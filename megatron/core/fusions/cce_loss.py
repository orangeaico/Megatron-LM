# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0
"""
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
    embeddings: torch.Tensor,          # [B, T, H]
    classifier_weight: torch.Tensor,   # [V, H] (or [V_local, H] with VP)
    labels: torch.Tensor,              # [B, T]
    vocab_size: int,
    impl: str = "cce",
    reduction: str = "none",
    shift: bool = True,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute (optionally shifted) per-token cross-entropy via CCE.
    Returns [B, T-1] if shift=True, else [B, T].
    """
    # --- Normalize shapes -------------------------------------------------
    # CCE expects embeddings[..., H] with embeddings.size()[:-1] == labels.size().
    # Megatron sometimes carries hidden states as [S, B, H]. If so, transpose
    # to [B, S, H] to match labels' [B, S].
    if embeddings.ndim == 3 and labels is not None and labels.ndim == 2:
        b0, s0, h = embeddings.size(0), embeddings.size(1), embeddings.size(2)
        lb, ls = labels.size(0), labels.size(1)
        # If embeddings are [S, B, H] and labels are [B, S], swap to [B, S, H].
        if b0 == ls and s0 == lb:
            embeddings = embeddings.transpose(0, 1).contiguous()
        # Optional: assert now aligned
        eb, es = embeddings.size(0), embeddings.size(1)
        if (eb, es) != (lb, ls):
            raise RuntimeError(
                f"CCE shape mismatch after normalization: "
                f"embeddings[...,:] has {embeddings.size()[:-1]} but labels have {labels.size()} "
                f"(expected embeddings.size()[:-1] == labels.size())."
            )
    if linear_cross_entropy is None:
        raise ImportError(
            "cut-cross-entropy is not installed but `use_linear_cross_entropy=True`.\n"
            "Install via: pip install \"cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git\"\n"
            f"Original import error: {_CCE_IMPORT_ERROR}"
        )

    vp_opts = _maybe_build_vp_opts(vocab_size)

    # CCE handles half/bfloat16 inputs and promotes to fp32 internally where needed.
    losses = linear_cross_entropy(
        embeddings,                      # [B, T, H]
        classifier_weight,               # [V, H] or [V_local, H]
        labels,                          # [B, T]
        impl=impl,
        reduction=reduction,
        shift=int(shift),
        ignore_index=ignore_index,
        vocab_parallel_options=vp_opts,
    )
    return losses