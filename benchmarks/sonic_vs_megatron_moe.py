#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare SonicMoE expert kernels against Megatron's TE grouped expert block.

This mirrors the useful part of sonic-moe's benchmarks/moe-cute.py, but adds a
Megatron TEGroupedMLP local expert path. Both paths use the same T/H/I/E/K shape,
weights, top-k expert assignments, and router probabilities.

What is timed:
  - Sonic local expert: TC_topk_router_metadata_triton + _UpProjection + _DownProjection.
  - Megatron local expert: TEGroupedMLP on already expert-sorted tokens.

What is intentionally not timed:
  - Router linear/top-k construction.
  - Megatron token permutation/combine around the local expert module.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Callable

import torch
import torch.nn.functional as F
from triton.testing import do_bench


def parse_thiek(value: str) -> tuple[int, int, int, int, int]:
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("--thiek must be T,H,I,E,K")
    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SonicMoE vs Megatron grouped MoE kernels")
    parser.add_argument("--thiek", type=parse_thiek, default=(65536, 2048, 768, 128, 8))
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--skip-correctness", action="store_true")
    return parser.parse_args()


def make_routing(
    x: torch.Tensor,
    num_experts: int,
    topk: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    router_w = torch.randn(num_experts, x.size(1), device=x.device, dtype=x.dtype) * 0.02
    logits = F.linear(x, router_w)
    topk_logits, topk_indices = logits.topk(topk, dim=-1)
    topk_scores = topk_logits.softmax(dim=-1, dtype=torch.float32)

    num_tokens = x.size(0)
    flat_experts = topk_indices.reshape(-1)
    sorted_experts, permutation = flat_experts.sort(stable=True)
    token_indices = (
        torch.arange(num_tokens, device=x.device, dtype=torch.long)
        .repeat_interleave(topk)
        .index_select(0, permutation)
    )
    sorted_scores = topk_scores.reshape(-1).index_select(0, permutation)
    tokens_per_expert = torch.bincount(flat_experts, minlength=num_experts).to(torch.int32)
    offsets = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)

    return (
        topk_scores,
        topk_indices.to(torch.int32),
        token_indices,
        sorted_scores,
        tokens_per_expert,
        offsets,
        router_w,
    )


def build_megatron_te_grouped_mlp(
    hidden_size: int,
    expert_hidden_size: int,
    num_experts: int,
    topk: int,
    dtype: torch.dtype,
):
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TERowParallelGroupedLinear,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=32 if hidden_size % 32 == 0 else 1,
        ffn_hidden_size=expert_hidden_size,
        num_moe_experts=num_experts,
        moe_ffn_hidden_size=expert_hidden_size,
        moe_router_topk=topk,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=False,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        params_dtype=dtype,
        perform_initialization=False,
        tensor_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
    )
    submodules = GroupedMLPSubmodules(
        linear_fc1=TEColumnParallelGroupedLinear,
        linear_fc2=TERowParallelGroupedLinear,
        activation_func=None,
    )
    pg_collection = ProcessGroupCollection(ep=None, expt_tp=None)
    module = TEGroupedMLP(num_experts, config, submodules, pg_collection)
    module.train()
    return module


def copy_megatron_te_weights(
    module,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
) -> None:
    """Copy [E, 2I, H] and [E, H, I] weights into TEGroupedMLP."""
    with torch.no_grad():
        for expert_idx in range(fc1_weight.size(0)):
            getattr(module.linear_fc1, f"weight{expert_idx}").copy_(fc1_weight[expert_idx])
            getattr(module.linear_fc2, f"weight{expert_idx}").copy_(fc2_weight[expert_idx])


def make_sonic_weights_from_te_weights(
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert TE chunked SwiGLU weights to Sonic's interleaved QuACK layout."""
    gate, up = fc1_weight.chunk(2, dim=1)
    fc1_interleaved = torch.stack((gate, up), dim=2).reshape_as(fc1_weight)
    fc1_backing = fc1_interleaved.transpose(1, 2).contiguous()
    fc2_backing = fc2_weight.transpose(1, 2).contiguous()
    return (
        fc1_backing.permute(2, 1, 0).detach().requires_grad_(),
        fc2_backing.permute(2, 1, 0).detach().requires_grad_(),
    )


def sonic_local_forward(
    x: torch.Tensor,
    topk_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    w1_sonic: torch.Tensor,
    w2_sonic: torch.Tensor,
    activation_type,
) -> torch.Tensor:
    from sonicmoe.functional import _DownProjection, _UpProjection
    from sonicmoe.functional.triton_kernels import TC_topk_router_metadata_triton

    num_tokens = x.size(0)
    num_experts = w2_sonic.size(-1)
    topk = topk_indices.size(1)
    total_expert_freq = num_tokens * topk
    device = x.device

    s_scatter_idx = torch.empty(total_expert_freq, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(total_expert_freq, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(num_experts, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(total_expert_freq, dtype=torch.int32, device=device)
    TC_topk_router_metadata_triton(
        topk_indices,
        num_experts,
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
    )

    a, h = _UpProjection.apply(
        x,
        w1_sonic,
        None,
        expert_frequency_offset,
        total_expert_freq,
        topk,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        None,
        False,
        activation_type,
        not torch.is_grad_enabled(),
        False,
    )
    return _DownProjection.apply(
        a,
        h,
        w2_sonic,
        None,
        topk_scores,
        expert_frequency_offset,
        num_tokens,
        topk,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        None,
        False,
        activation_type,
    )


def megatron_grouped_local_forward(
    module,
    x_permuted: torch.Tensor,
    sorted_scores: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    output, _ = module(x_permuted, tokens_per_expert, sorted_scores)
    return output


def bench_forward(name: str, fn: Callable[[], torch.Tensor], warmup: int, repeats: int) -> float:
    time.sleep(0.2)
    torch.cuda.synchronize()
    timing = do_bench(fn, warmup=warmup, rep=repeats)
    print(f"{name} Fwd Average time: {timing:.3f} ms")
    return timing


def bench_forward_backward(
    name: str,
    fn: Callable[[], tuple[torch.Tensor, list[torch.Tensor]]],
    grad_output: torch.Tensor,
    warmup: int,
    repeats: int,
) -> float:
    def step():
        out, grad_tensors = fn()
        out.backward(grad_output, retain_graph=False)
        for tensor in grad_tensors:
            tensor.grad = None

    time.sleep(0.2)
    torch.cuda.synchronize()
    timing = do_bench(step, warmup=warmup, rep=repeats)
    print(f"{name} Fwd + Bwd Average time: {timing:.3f} ms")
    return timing


def main() -> None:
    args = parse_args()
    os.environ["MOE_EXPERT_LOG_TIMING"] = "0"
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    T, H, I, E, K = args.thiek
    device = torch.device("cuda")

    from sonicmoe.enums import ActivationType

    activation_type = ActivationType.SWIGLU

    torch.manual_seed(1111)
    torch.cuda.manual_seed_all(1111)

    x = (0.2 * torch.randn(T, H, device=device, dtype=torch_dtype)).requires_grad_()
    (
        topk_scores,
        topk_indices,
        token_indices,
        sorted_scores,
        tokens_per_expert,
        _,
        _,
    ) = make_routing(x, E, K)
    topk_scores = topk_scores.detach().requires_grad_()
    sorted_scores = sorted_scores.detach().requires_grad_()

    te_mlp = build_megatron_te_grouped_mlp(H, I, E, K, torch_dtype)
    fc1_weight = 0.02 * torch.randn(E, 2 * I, H, device=device, dtype=torch_dtype)
    fc2_weight = 0.02 * torch.randn(E, H, I, device=device, dtype=torch_dtype)
    copy_megatron_te_weights(te_mlp, fc1_weight, fc2_weight)
    w1_sonic, w2_sonic = make_sonic_weights_from_te_weights(fc1_weight, fc2_weight)
    x_permuted = x.detach().index_select(0, token_indices).requires_grad_()
    te_grad_tensors = [x_permuted, sorted_scores, *te_mlp.parameters()]

    print(f"T {T}, I {I}, H {H}, E {E}, K {K}, dtype {args.dtype}, activation swiglu")
    print("Sonic path: TC_topk_router_metadata_triton + _UpProjection + _DownProjection")
    print("Megatron path: TEGroupedMLP on sorted tokens")

    if not args.skip_correctness:
        with torch.enable_grad():
            sonic_out = sonic_local_forward(
                x, topk_scores, topk_indices, w1_sonic, w2_sonic, activation_type
            )
            megatron_permuted_out = megatron_grouped_local_forward(
                te_mlp, x_permuted, sorted_scores, tokens_per_expert
            )
            megatron_out = torch.zeros_like(sonic_out)
            megatron_out.index_add_(0, token_indices, megatron_permuted_out)
            diff = (sonic_out.float() - megatron_out.float()).abs()
            print(f"max abs diff on output: {diff.max().item():.6f}")
            print(f"mean abs diff on output: {diff.mean().item():.6f}")

    def sonic_fwd():
        return sonic_local_forward(x, topk_scores, topk_indices, w1_sonic, w2_sonic, activation_type)

    def megatron_fwd():
        return megatron_grouped_local_forward(
            te_mlp, x_permuted, sorted_scores, tokens_per_expert
        )

    sonic_fwd_ms = bench_forward("Sonic local", sonic_fwd, args.warmup, args.repeats)
    megatron_fwd_ms = bench_forward(
        "Megatron TE grouped local", megatron_fwd, args.warmup, args.repeats
    )

    sonic_dout = torch.randn(T, H, device=device, dtype=torch_dtype)
    megatron_dout = torch.randn(T * K, H, device=device, dtype=torch_dtype)

    sonic_e2e_ms = bench_forward_backward(
        "Sonic local",
        lambda: (
            sonic_local_forward(x, topk_scores, topk_indices, w1_sonic, w2_sonic, activation_type),
            [x, topk_scores, w1_sonic, w2_sonic],
        ),
        sonic_dout,
        args.warmup,
        args.repeats,
    )
    megatron_e2e_ms = bench_forward_backward(
        "Megatron TE grouped local",
        lambda: (
            megatron_grouped_local_forward(
                te_mlp, x_permuted, sorted_scores, tokens_per_expert
            ),
            te_grad_tensors,
        ),
        megatron_dout,
        args.warmup,
        args.repeats,
    )

    print(f"Sonic local Bwd Average time: {sonic_e2e_ms - sonic_fwd_ms:.3f} ms")
    print(f"Megatron TE grouped local Bwd Average time: {megatron_e2e_ms - megatron_fwd_ms:.3f} ms")
    print(f"Fwd speedup, Megatron/Sonic: {megatron_fwd_ms / sonic_fwd_ms:.3f}x")
    print(f"Fwd+Bwd speedup, Megatron/Sonic: {megatron_e2e_ms / sonic_e2e_ms:.3f}x")


if __name__ == "__main__":
    main()
