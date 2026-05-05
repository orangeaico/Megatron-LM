#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare SonicMoE expert kernels against a Megatron-style grouped-mm expert block.

This mirrors the useful part of sonic-moe's benchmarks/moe-cute.py, but adds a
Megatron-style local expert path. Both paths use the same T/H/I/E/K shape,
weights, top-k expert assignments, and router probabilities.

What is timed:
  - Sonic local expert: TC_topk_router_metadata_triton + _UpProjection + _DownProjection.
  - Megatron grouped local expert: grouped_mm fc1 + SwiGLU/prob scale + grouped_mm fc2
    on already expert-sorted tokens.

What is intentionally not timed:
  - Router linear/top-k construction.
  - Megatron token permutation/combine around the local expert module.
"""

from __future__ import annotations

import argparse
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


def resolve_grouped_mm() -> Callable:
    if hasattr(F, "grouped_mm"):
        return lambda x, w, offsets: F.grouped_mm(x, w, offs=offsets)
    if hasattr(torch, "_grouped_mm"):
        return lambda x, w, offsets: torch._grouped_mm(x, w, offsets)
    raise RuntimeError(
        "No grouped_mm implementation found. Use a PyTorch build with "
        "torch.nn.functional.grouped_mm or torch._grouped_mm."
    )


def swiglu_interleaved(h: torch.Tensor) -> torch.Tensor:
    gate = h[..., 0::2]
    up = h[..., 1::2]
    return up * F.silu(gate)


def make_routing(
    x: torch.Tensor,
    num_experts: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    return topk_scores, topk_indices.to(torch.int32), token_indices, sorted_scores, offsets, router_w


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
    grouped_mm: Callable,
    x_permuted: torch.Tensor,
    sorted_scores: torch.Tensor,
    offsets: torch.Tensor,
    w1_megatron: torch.Tensor,
    w2_megatron: torch.Tensor,
) -> torch.Tensor:
    fc1 = grouped_mm(x_permuted, w1_megatron.transpose(1, 2), offsets)
    act = swiglu_interleaved(fc1) * sorted_scores.unsqueeze(-1)
    return grouped_mm(act, w2_megatron.transpose(1, 2), offsets)


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
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    T, H, I, E, K = args.thiek
    device = torch.device("cuda")

    from sonicmoe.enums import ActivationType

    activation_type = ActivationType.SWIGLU
    grouped_mm = resolve_grouped_mm()

    torch.manual_seed(1111)
    torch.cuda.manual_seed_all(1111)

    x = (0.2 * torch.randn(T, H, device=device, dtype=torch_dtype)).requires_grad_()
    topk_scores, topk_indices, token_indices, sorted_scores, offsets, _ = make_routing(x, E, K)
    topk_scores = topk_scores.detach().requires_grad_()
    sorted_scores = sorted_scores.detach().requires_grad_()

    # Sonic uses [2I, H, E] and [H, I, E]. Interleaved GLU rows are [gate0, up0, ...].
    w1_sonic = (0.02 * torch.randn(2 * I, H, E, device=device, dtype=torch_dtype)).requires_grad_()
    w2_sonic = (0.02 * torch.randn(H, I, E, device=device, dtype=torch_dtype)).requires_grad_()

    # Megatron grouped-mm baseline uses [E, out, in] expert weights.
    w1_megatron = w1_sonic.permute(2, 0, 1).contiguous().detach().requires_grad_()
    w2_megatron = w2_sonic.permute(2, 0, 1).contiguous().detach().requires_grad_()
    x_permuted = x.detach().index_select(0, token_indices).requires_grad_()

    print(f"T {T}, I {I}, H {H}, E {E}, K {K}, dtype {args.dtype}, activation swiglu")
    print("Sonic path: TC_topk_router_metadata_triton + _UpProjection + _DownProjection")
    print("Megatron path: grouped_mm fc1 + SwiGLU/prob scale + grouped_mm fc2 on sorted tokens")

    if not args.skip_correctness:
        with torch.enable_grad():
            sonic_out = sonic_local_forward(
                x, topk_scores, topk_indices, w1_sonic, w2_sonic, activation_type
            )
            megatron_permuted_out = megatron_grouped_local_forward(
                grouped_mm, x_permuted, sorted_scores, offsets, w1_megatron, w2_megatron
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
            grouped_mm, x_permuted, sorted_scores, offsets, w1_megatron, w2_megatron
        )

    sonic_fwd_ms = bench_forward("Sonic local", sonic_fwd, args.warmup, args.repeats)
    megatron_fwd_ms = bench_forward("Megatron grouped local", megatron_fwd, args.warmup, args.repeats)

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
        "Megatron grouped local",
        lambda: (
            megatron_grouped_local_forward(
                grouped_mm, x_permuted, sorted_scores, offsets, w1_megatron, w2_megatron
            ),
            [x_permuted, sorted_scores, w1_megatron, w2_megatron],
        ),
        megatron_dout,
        args.warmup,
        args.repeats,
    )

    print(f"Sonic local Bwd Average time: {sonic_e2e_ms - sonic_fwd_ms:.3f} ms")
    print(f"Megatron grouped local Bwd Average time: {megatron_e2e_ms - megatron_fwd_ms:.3f} ms")
    print(f"Fwd speedup, Megatron/Sonic: {megatron_fwd_ms / sonic_fwd_ms:.3f}x")
    print(f"Fwd+Bwd speedup, Megatron/Sonic: {megatron_e2e_ms / sonic_e2e_ms:.3f}x")


if __name__ == "__main__":
    main()
