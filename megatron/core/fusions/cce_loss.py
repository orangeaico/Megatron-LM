# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0
"""
from __future__ import annotations
Thin wrapper that adapts Megatron-LM's tensor-parallel vocab sharding to
apple/ml-cross-entropy's `linear_cross_entropy` API.

Install:
  pip install "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"
Docs:
  https://github.com/apple/ml-cross-entropy
"""
from typing import Optional
import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_group,
)
from megatron.core.tensor_parallel.utils import VocabUtility

try:
    # Upstream API
    from cut_cross_entropy import linear_cross_entropy, VocabParallelOptions  # type: ignore
except Exception as e:  # pragma: no cover
    linear_cross_entropy = None
    VocabParallelOptions = None
    _CCE_IMPORT_ERROR = e
else:
    _CCE_IMPORT_ERROR = None


def _maybe_build_vp_opts(vocab_size: int) -> Optional["VocabParallelOptions"]:
    """Create VocabParallelOptions when TP>1; otherwise return None."""
    tp_world = get_tensor_model_parallel_world_size()
    if tp_world <= 1:
        return None
    tp_rank = get_tensor_model_parallel_rank()
    group = get_tensor_model_parallel_group()
    start, end = VocabUtility.vocab_range_from_global_vocab_size(vocab_size, tp_rank, tp_world)
    return VocabParallelOptions(start, end, group=group)


def cce_per_token_loss(
    *,
    embeddings: torch.Tensor,          # [B, T, H] or [T, B, H] or [B, T_local, H] with SP
    classifier_weight: torch.Tensor,   # [V, H] (or [V_local, H] with VP)
    labels: torch.Tensor,              # [B, T_global] (Megatron's labels are global wrt SP)
    vocab_size: int,
    impl: str = "cce",
    reduction: str = "none",
    shift: bool = True,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute (optionally shifted) per-token cross-entropy via CCE.
    Returns [B, T-1] if shift=True, else [B, T].
    """
    tp_world = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    # Track whether we narrowed labels for sequence parallel and the slice range.
    sp_sliced = False
    sp_start: int | None = None
    sp_end: int | None = None
    # We'll keep a local copy to hand to CCE, and preserve the global length for padding.
    labels_local = labels
    # --- Normalize shapes -------------------------------------------------
    # CCE expects embeddings[..., H] with embeddings.size()[:-1] == labels.size().
    if embeddings.ndim == 3 and labels is not None and labels.ndim == 2:
        B_lab, S_glob_global = labels.size(0), labels.size(1)
        dim0, dim1, _ = embeddings.size()

        # Case A: embeddings already [B, S, H] (no SP or gathered SP)
        if dim0 == B_lab and dim1 == S_glob_global:
            pass  # ok

        # Case B: embeddings [S, B, H] (no SP)
        elif dim0 == S_glob_global and dim1 == B_lab:
            embeddings = embeddings.transpose(0, 1).contiguous()  # -> [B, S, H]

        else:
            # Possibly Sequence Parallel (SP): embeddings carry only a local slice of S.
            # Two layouts to handle:
            #  - [S_local, B, H]  (dim0=S_local, dim1=B)
            #  - [B, S_local, H]  (dim0=B, dim1=S_local)
            # In SP, S_glob = S_local * tp_world, and we must also slice labels to the local range.
            if tp_world > 1:
                # Try layout [S_local, B, H]
                if dim1 == B_lab and (S_glob_global % dim0 == 0) and (S_glob_global // dim0 == tp_world):
                    S_local = dim0
                    sp_start = tp_rank * S_local
                    sp_end = sp_start + S_local
                    # transpose to [B, S_local, H] and slice labels to local window
                    embeddings = embeddings.transpose(0, 1).contiguous()
                    labels_local = labels[:, sp_start:sp_end]
                    sp_sliced = True
                # Try layout [B, S_local, H]
                elif dim0 == B_lab and (S_glob_global % dim1 == 0) and (S_glob_global // dim1 == tp_world):
                    S_local = dim1
                    sp_start = tp_rank * S_local
                    sp_end = sp_start + S_local
                    labels_local = labels[:, sp_start:sp_end]
                    sp_sliced = True
                else:
                    raise RuntimeError(
                        "CCE: could not reconcile embeddings and labels shapes under sequence parallel.\n"
                        f"embeddings[...,:]={tuple(embeddings.size()[:-1])}, labels={tuple(labels.size())}, "
                        f"tp_world={tp_world}"
                    )

        # Final check after normalization/slicing
        if embeddings.size(0) != labels_local.size(0) or embeddings.size(1) != labels_local.size(1):
            raise RuntimeError(
                f"CCE shape mismatch after normalization: embeddings[...,:] has {embeddings.size()[:-1]} "
                f"but labels_local have {labels_local.size()} (expected embeddings.size()[:-1] == labels_local.size())."
            )
    if linear_cross_entropy is None:
        raise ImportError(
            "cut-cross-entropy is not installed but `use_linear_cross_entropy=True`.\n"
            "Install via: pip install \"cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git\"\n"
            f"Original import error: {_CCE_IMPORT_ERROR}"
        )

    # Build VP options (and assert shard size matches)
    vp_opts = _maybe_build_vp_opts(vocab_size)

    # CCE handles half/bfloat16 inputs and promotes to fp32 internally where needed.
    losses = linear_cross_entropy(
        embeddings,                      # [B, T or T_local, H]
        classifier_weight,               # [V, H] or [V_local, H]
        labels_local,                    # [B, T or T_local]
        impl=impl,
        reduction=reduction,
        shift=int(shift),
        ignore_index=ignore_index,
        vocab_parallel_options=vp_opts,
    )

    # --- SP padding to global length ---------------------------------------
    # Megatron's loss_func multiplies with a *global* loss_mask [B, S_global].
    # If we sliced labels/embeddings for SP, zero-pad losses back to [B, S_global]
    # so shapes match and masked sum stays correct.
    if labels is not None and labels.ndim == 2:
        # Use the ORIGINAL global length (before slicing) for padding.
        B_lab, S_glob_global = labels.size(0), S_glob_global
        # If we performed SP slicing above, we recorded (sp_start, sp_end).
        if sp_sliced and sp_start is not None and sp_end is not None:
            # CCE may return [B, S_local] or [B, S_local-1] if shift=True.
            width = losses.size(1)
            # Clamp end to start + width to handle shift=True transparently.
            end_eff = sp_start + width
            out = torch.zeros(B_lab, S_glob_global, device=losses.device, dtype=losses.dtype)
            out[:, sp_start:end_eff] = losses
            losses = out

    return losses