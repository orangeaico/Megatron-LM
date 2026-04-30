# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0
"""
Thin wrapper that adapts Megatron-LM's tensor-parallel vocab sharding to
apple/ml-cross-entropy's `linear_cross_entropy` API.
"""

from typing import Optional

import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
)

try:
    from cut_cross_entropy import VocabParallelOptions, linear_cross_entropy  # type: ignore
except Exception as e:  # pragma: no cover
    linear_cross_entropy = None
    VocabParallelOptions = None
    _CCE_IMPORT_ERROR = e
else:
    _CCE_IMPORT_ERROR = None


def _maybe_build_vp_opts(vocab_size: int) -> Optional["VocabParallelOptions"]:
    tp_world = get_tensor_model_parallel_world_size()
    if tp_world <= 1:
        return None
    return VocabParallelOptions.from_vocab(vocab_size, group=get_tensor_model_parallel_group())


def cce_per_token_loss(
    *,
    embeddings: torch.Tensor,
    classifier_weight: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    impl: str = "torch_compile",
    reduction: str = "none",
    shift: bool = False,
    ignore_index: int = -100,
    return_lse: bool = False,
    temperature: float = 1.0,
) -> torch.Tensor:
    embeddings = embeddings.transpose(0, 1).contiguous()
    if embeddings.size(0) != labels.size(0) or embeddings.size(1) != labels.size(1):
        raise RuntimeError(
            f"CCE shape mismatch after normalization: embeddings[...,:] has "
            f"{embeddings.size()[:-1]} but labels have {labels.size()}."
        )
    if linear_cross_entropy is None:
        raise ImportError(
            "cut-cross-entropy is not installed but `use_linear_cross_entropy=True`.\n"
            "Install via: pip install "
            "\"cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git\"\n"
            f"Original import error: {_CCE_IMPORT_ERROR}"
        )

    return linear_cross_entropy(
        embeddings,
        classifier_weight,
        labels,
        impl=impl,
        reduction=reduction,
        shift=int(shift),
        ignore_index=ignore_index,
        vocab_parallel_options=_maybe_build_vp_opts(vocab_size),
        return_lse=return_lse,
        softcap=temperature,
    )
