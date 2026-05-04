# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from math import ceil
from typing import Optional, Protocol, Tuple

import torch
import torch.nn.functional as F

from megatron.core import tensor_parallel
from megatron.core.activations import squared_relu
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, ShardedTensorFactory
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fusions.fused_bias_geglu import quick_gelu, weighted_bias_quick_geglu_impl
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl
from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.transformer.mlp import (
    MLP,
    MLPSubmodules,
    TEActivationFunctionBuilder,
    apply_swiglu_sharded_factory,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    ProcessGroupCollection,
    get_align_size_for_quantization,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    sharded_state_dict_default,
)
from megatron.core.typed_torch import apply_module, not_none
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import Fp8Padding, Fp8Unpadding
else:
    Fp8Padding, Fp8Unpadding = None, None

try:
    import flashinfer.fused_moe as fused_moe
    from flashinfer.fused_moe.core import ActivationType

    HAVE_FLASHINFER = True
except ImportError:
    HAVE_FLASHINFER = False

from megatron.core.inference.moe import ActivationType as McoreActivationType
from megatron.core.inference.moe import (
    InferenceGroupedGemmBackend,
    mcore_fused_moe,
    resolve_inference_grouped_gemm_backend,
)

logger = logging.getLogger(__name__)


class GroupedLinearFc1Interface(Protocol):
    """Interface for linear_fc1 module in TEGroupedMLP."""

    def forward(
        self, permuted_local_hidden_states: torch.Tensor, tokens_per_expert: list[int], /
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward method for linear_fc1 module."""
        ...

    def backward_dw(self) -> None:
        """Backward method for linear_fc1 module."""
        ...


class GroupedLinearFc1Builder(Protocol):
    """Protocol describing how to build a linear_fc1 layer in TEGroupedMLP."""

    def __call__(
        self,
        num_local_experts: int,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str | None,
        pg_collection: ProcessGroupCollection | None,
    ) -> GroupedLinearFc1Interface:
        """Builds a linear_fc1 layer for TEGroupedMLP."""
        ...


class GroupedLinearFc2Interface(Protocol):
    """Protocol for linear_fc2 module in TEGroupedMLP."""

    def forward(
        self, intermediate_parallel: torch.Tensor, tokens_per_expert: list[int], /
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward method for linear_fc2 module."""
        ...

    def backward_dw(self) -> None:
        """Backward method for linear_fc2 module."""
        ...


class GroupedLinearFc2Builder(Protocol):
    """Protocol describing how to build a linear_fc2 layer in TEGroupedMLP."""

    def __call__(
        self,
        num_local_experts: int,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str | None,
        pg_collection: ProcessGroupCollection | None,
    ) -> GroupedLinearFc2Interface:
        """Builds a linear_fc2 layer for TEGroupedMLP."""
        ...


@dataclass
class GroupedMLPSubmodules:
    """
    The dataclass for ModuleSpecs of TEGroupedMLP submodules
    including  linear fc1, activation function, linear fc2.
    """

    linear_fc1: GroupedLinearFc1Builder

    linear_fc2: GroupedLinearFc2Builder

    activation_func: TEActivationFunctionBuilder | None = None
    """
    Builder for an activation function module; only used if config.use_te_activation_func is True.
    """


class TEGroupedMLP(MegatronModule):
    """An efficient implementation of the Experts layer using TE's GroupedLinear.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    # TODO(M4): breaking api, switched from pass in tp_group to pass in pg_collection.
    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        submodules: GroupedMLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        self.input_size = self.config.hidden_size
        assert not (
            self.config.add_bias_linear and config.bias_dropout_fusion
        ), "bias_dropout_fusion is not supported in TEGroupedMLP when add_bias_linear=True"

        self.ep_group = pg_collection.ep
        self.tp_group = pg_collection.expt_tp

        # Double the output width with gated linear unit, see https://arxiv.org/pdf/2002.05202.pdf
        ffn_hidden_size = not_none(self.config.moe_ffn_hidden_size)
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.linear_fc1 = submodules.linear_fc1(
            self.num_local_experts,
            self.input_size if self.config.moe_latent_size is None else self.config.moe_latent_size,
            ffn_hidden_size,
            config=self.config,
            init_method=not_none(self.config.init_method),
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=True,
            tp_comm_buffer_name='fc1',
            pg_collection=pg_collection,
        )

        if self.config.use_te_activation_func and not (submodules.activation_func is None):
            self.activation_func = apply_module(submodules.activation_func(config=self.config))
        else:
            self.activation_func = self.config.activation_func

        self.linear_fc2 = submodules.linear_fc2(
            self.num_local_experts,
            not_none(self.config.moe_ffn_hidden_size),
            (
                self.config.hidden_size
                if self.config.moe_latent_size is None
                else self.config.moe_latent_size
            ),
            config=self.config,
            init_method=not_none(self.config.output_layer_init_method),
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=True,
            tp_comm_buffer_name='fc2',
            pg_collection=pg_collection,
        )

        self.offload_expert_fc1 = (
            self.config.fine_grained_activation_offloading
            and "expert_fc1" in self.config.offload_modules
        )

        self.offload_moe_act = (
            self.config.fine_grained_activation_offloading
            and "moe_act" in self.config.offload_modules
        )

        self.activation_recompute = (
            self.config.recompute_granularity == 'selective'
            and "moe_act" in self.config.recompute_modules
        )
        if self.activation_recompute and (self.config.fp8 or self.config.fp4):
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.linear_fc2)

        # This is to avoid the CPU overhead of multiple d2h copies
        if self.offload_expert_fc1:
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.linear_fc1)

        if self.config.fp8 or self.config.fp4:
            assert HAVE_TE, "FP8 and FP4 requires TE."
            self.quantization_padding = Fp8Padding(self.num_local_experts)
            self.quantization_unpadding = Fp8Unpadding(self.num_local_experts)

    @staticmethod
    def _apply_bias(intermediate_parallel, bias_parallel, tokens_per_expert, permuted_probs):
        if bias_parallel is None:
            return intermediate_parallel
        shape = intermediate_parallel.shape
        return (
            torch.cat(
                [
                    t + b * p
                    for t, b, p in zip(
                        torch.split(intermediate_parallel.view(-1, shape[-1]), tokens_per_expert),
                        bias_parallel,
                        torch.split(permuted_probs, tokens_per_expert),
                    )
                ]
            )
            .view(shape)
            .to(intermediate_parallel.dtype)
        )

    def bias_act_func(self, intermediate_parallel, bias_parallel, permuted_probs):
        """
        Applies bias and activation function to the output of linear_fc1.
        """
        if self.config.use_te_activation_func:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)
            if permuted_probs is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * permuted_probs
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        elif self.config.bias_activation_fusion:
            if self.activation_func == F.silu and self.config.gated_linear_unit:
                # dtype is handled inside the fused kernel
                intermediate_parallel = weighted_bias_swiglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    permuted_probs,
                    self.config.activation_func_fp8_input_store,
                )
            elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                intermediate_parallel = weighted_bias_quick_geglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    permuted_probs,
                    self.config.activation_func_fp8_input_store,
                    self.config.glu_linear_offset,
                    self.config.activation_func_clamp_value,
                )
            else:
                raise ValueError("Only support fusion of swiglu and quick_gelu in TEGroupedMLP.")
        elif self.activation_func == squared_relu and self.config.use_fused_weighted_squared_relu:
            assert bias_parallel is None, "Bias is not supported with fused weighted squared relu."
            intermediate_parallel = weighted_squared_relu_impl(
                intermediate_parallel, permuted_probs
            )
        else:
            if self.config.gated_linear_unit:

                def glu(x):
                    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                    if (val := self.config.activation_func_clamp_value) is not None:
                        x_glu = x_glu.clamp(min=None, max=val)
                        x_linear = x_linear.clamp(min=-val, max=val)
                    return self.config.activation_func(x_glu) * (
                        x_linear + self.config.glu_linear_offset
                    )

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)
            original_dtype = intermediate_parallel.dtype
            intermediate_parallel = intermediate_parallel * permuted_probs
            intermediate_parallel = intermediate_parallel.to(original_dtype)
        return intermediate_parallel

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of TEGroupedMLP

        Args:
            permuted_local_hidden_states (torch.Tensor): The permuted input hidden states of the
            local experts.
            tokens_per_expert (torch.Tensor): The number of tokens per expert.
            permuted_probs (torch.Tensor): The permuted probs of each token produced by the router.

        Return:
            output (torch.Tensor): The output of the local experts.
        """
        tokens_per_expert: list[int] = tokens_per_expert.tolist()
        if self.config.fp8 or self.config.fp4:
            actual_tokens_per_expert = tokens_per_expert
            permuted_local_hidden_states, tokens_per_expert = self.quantization_padding(
                permuted_local_hidden_states, tokens_per_expert
            )
            permuted_probs, _ = self.quantization_padding(
                permuted_probs.unsqueeze(-1), actual_tokens_per_expert
            )
        else:
            permuted_probs = permuted_probs.unsqueeze(-1)

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs * permuted_local_hidden_states
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        with off_interface(
            self.offload_expert_fc1, permuted_local_hidden_states, "expert_fc1"
        ) as permuted_local_hidden_states:
            fc1_output, bias_parallel = apply_module(self.linear_fc1)(
                permuted_local_hidden_states, tokens_per_expert
            )
        if self.offload_expert_fc1:
            fc1_output = off_interface.group_commit(
                fc1_output,
                name="expert_fc1",
                forced_released_tensors=[permuted_local_hidden_states],
            )

        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with off_interface(self.offload_moe_act, fc1_output, "moe_act") as fc1_output:
                bias_act_output = self.activation_checkpoint.checkpoint(
                    self.bias_act_func, fc1_output, bias_parallel, permuted_probs
                )
        else:
            with off_interface(self.offload_moe_act, fc1_output, "moe_act") as fc1_output:
                bias_act_output = self.bias_act_func(fc1_output, bias_parallel, permuted_probs)
        output, output_bias = apply_module(self.linear_fc2)(bias_act_output, tokens_per_expert)
        if self.activation_recompute:
            self.activation_checkpoint.discard_output_and_register_recompute(output)

        # Delay the offload of the moe act until after the linear_fc2 has been computed
        # to make sure the fc1_output is reloaded to GPU before recomputing moe_act.
        if self.offload_moe_act:
            output = off_interface.group_commit(
                output, name="moe_act", forced_released_tensors=[fc1_output]
            )
        output = self._apply_bias(output, output_bias, tokens_per_expert, permuted_probs)

        # upad and concat the output
        if self.config.fp8 or self.config.fp4:
            output = self.quantization_unpadding(output, actual_tokens_per_expert)

        output_bias = None

        return output, output_bias

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Maps local expert to global experts.
        The sharded state dict is interchangable with SequentialMLP's.
        """
        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        sharded_state_dict = {}
        for name, module in self._modules.items():
            sub_sd = sharded_state_dict_default(
                module, f'{name}.', sharded_offsets, metadata, tp_group=self.tp_group
            )
            if name == 'linear_fc1' and self.config.gated_linear_unit:
                num_global_experts = self.ep_group.size() * self.num_local_experts
                local_expert_indices_offset = self.ep_group.rank() * self.num_local_experts
                ep_axis = len(sharded_offsets)
                for i in range(self.num_local_experts):
                    if singleton_local_shards:
                        new_sharded_offsets = sharded_offsets
                    else:
                        new_sharded_offsets = (
                            *sharded_offsets,
                            (ep_axis, local_expert_indices_offset + i, num_global_experts),
                        )
                    for k in (f'{name}.weight{i}', f'{name}.bias{i}'):
                        if k in sub_sd:
                            sub_sd[k] = apply_swiglu_sharded_factory(
                                sub_sd[k], new_sharded_offsets, singleton_local_shards
                            )
            if singleton_local_shards:
                replace_prefix_for_sharding(sub_sd, '', f'{prefix}experts.')
            else:
                # Add prefix here to match sequential's keys
                replace_prefix_for_sharding(sub_sd, f'{name}.', f'{prefix}experts.{name}.')
            sharded_state_dict.update({f"{prefix}{k}": v for k, v in sub_sd.items()})
        return sharded_state_dict

    def backward_dw(self):
        """Performs backward pass for weight gradients in TEGroupedMLP.

        This method executes the backward pass for weight gradients by calling
        backward_dw() on the linear layers in reverse order (fc2 followed by fc1).
        If an error occurs during execution, it is caught and re-raised with a
        descriptive message.
        """
        self.linear_fc2.backward_dw()
        self.linear_fc1.backward_dw()


class SonicGroupedMLP(MegatronModule):
    """MoE experts backed by SonicMoE local expert kernels.

    This intentionally keeps Megatron's router, token dispatcher, expert
    parallelism, and combine path unchanged. SonicMoE only replaces the local
    grouped expert MLP compute after tokens have already been dispatched to this
    rank's local experts.

    Unlike TEGroupedMLP, this module stores expert weights in a layout that can
    be viewed as Sonic's [2I_local, H, E] and [H, I_local, E] kernel inputs with
    stride(0) == 1. It exposes the old per-expert grouped-linear checkpoint
    layout through state-dict factories. This avoids materializing
    torch.stack([weight0, ..., weightN]) every forward while keeping checkpoint
    compatibility with the default grouped expert implementation.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        submodules: Optional[GroupedMLPSubmodules] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        del submodules
        self.num_local_experts = num_local_experts
        self.input_size = self.config.hidden_size
        self.ep_group = pg_collection.ep
        self.tp_group = pg_collection.expt_tp
        self.expert_parallel = self.config.expert_model_parallel_size > 1
        self._validate_sonic_config()

        ffn_hidden_size = not_none(self.config.moe_ffn_hidden_size)
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        tp_size = self.tp_group.size()
        fc1_output_size = ffn_hidden_size // tp_size
        fc2_input_size = not_none(self.config.moe_ffn_hidden_size) // tp_size

        self.linear_fc1_weight = torch.nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.input_size,
                fc1_output_size,
                dtype=self.config.params_dtype,
            )
        )
        self.linear_fc2_weight = torch.nn.Parameter(
            torch.empty(
                self.num_local_experts,
                fc2_input_size,
                self.input_size,
                dtype=self.config.params_dtype,
            )
        )

        not_none(self.config.init_method)(self.linear_fc1_weight)
        not_none(self.config.output_layer_init_method)(self.linear_fc2_weight)

        for param in self.parameters():
            setattr(param, "allreduce", not self.expert_parallel)
        setattr(self.linear_fc1_weight, "partition_dim", 2)
        setattr(self.linear_fc1_weight, "partition_stride", 1)
        setattr(self.linear_fc2_weight, "partition_dim", 1)
        setattr(self.linear_fc2_weight, "partition_stride", 1)

        self._logged_routed_forward = False
        self._register_load_state_dict_pre_hook(self._merge_legacy_sonic_state_dict_keys)

    @staticmethod
    def _interleave_glu_weight(weight: torch.Tensor) -> torch.Tensor:
        """Convert [gate..., up...] GLU columns to Sonic [gate0, up0, ...] columns."""
        *prefix, two_intermediate = weight.shape
        intermediate = two_intermediate // 2
        return (
            weight.reshape(*prefix, 2, intermediate)
            .transpose(-1, -2)
            .reshape(*prefix, two_intermediate)
            .contiguous()
        )

    @staticmethod
    def _deinterleave_glu_weight(weight: torch.Tensor) -> torch.Tensor:
        """Convert Sonic [gate0, up0, ...] GLU columns to [gate..., up...] columns."""
        *prefix, two_intermediate = weight.shape
        intermediate = two_intermediate // 2
        return (
            weight.reshape(*prefix, intermediate, 2)
            .transpose(-1, -2)
            .reshape(*prefix, two_intermediate)
            .contiguous()
        )

    def _validate_sonic_config(self):
        if self.config.add_bias_linear:
            raise ValueError("SonicGroupedMLP does not support expert bias.")
        if self.config.fp8 or self.config.fp4:
            raise ValueError("SonicGroupedMLP v1 supports only BF16/FP16 unquantized experts.")
        if self.config.moe_latent_size is not None:
            raise ValueError("SonicGroupedMLP does not support MoE latent projections.")
        if not self.config.gated_linear_unit:
            raise ValueError("SonicGroupedMLP requires a gated linear unit activation.")
        if self.config.activation_func not in (F.silu, F.gelu):
            raise ValueError("SonicGroupedMLP supports SwiGLU and GeGLU only.")
        if self.config.hidden_size < 512 or self.config.hidden_size % 64 != 0:
            raise ValueError("SonicGroupedMLP requires hidden_size >= 512 and divisible by 64.")
        if self.config.moe_router_topk > 16:
            raise ValueError("SonicGroupedMLP requires moe_router_topk <= 16.")

        ffn_hidden_size = not_none(self.config.moe_ffn_hidden_size)
        tp_size = self.tp_group.size()
        if ffn_hidden_size % tp_size != 0:
            raise ValueError("moe_ffn_hidden_size must be divisible by expert tensor parallel size.")
        if (ffn_hidden_size // tp_size) % 64 != 0:
            raise ValueError(
                "SonicGroupedMLP requires per-ETP moe_ffn_hidden_size to be divisible by 64."
            )

    def _sonic_activation_type(self):
        if self.config.activation_func == F.silu:
            return "swiglu"
        if self.config.activation_func == F.gelu:
            return "geglu"
        raise ValueError("Unsupported SonicGroupedMLP activation function.")

    def _sonic_weights(self):
        # Sonic expects [out, in, E] with stride(0) == 1. Contiguous backing
        # storage [E, in, out] gives that view without a per-forward copy.
        # linear_fc1_weight is stored in Sonic's interleaved GLU order.
        return (
            self.linear_fc1_weight.permute(2, 1, 0),
            self.linear_fc2_weight.permute(2, 1, 0),
        )

    def _legacy_fc1_weight(self):
        return self._deinterleave_glu_weight(self.linear_fc1_weight)

    def _zero_token_forward(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        output = hidden_states.new_empty(hidden_states.shape)
        zero = self.linear_fc1_weight.sum() * 0 + self.linear_fc2_weight.sum() * 0
        return output + zero, None

    def can_use_routed_forward(self) -> bool:
        """Return whether Sonic can own local gather/aggregate for this rank."""
        return self.ep_group.size() == 1 and self.tp_group.size() == 1

    def forward_from_routing(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward using Sonic's routing-aware local expert path.

        This bypasses Megatron's local token permutation and lets Sonic gather
        tokens inside the up-projection GEMM and aggregate down-projection
        results back to token order.
        """
        if not self.can_use_routed_forward():
            raise RuntimeError(
                "SonicGroupedMLP routed forward currently supports only EP=1 and "
                "expert TP=1. Use the standard Sonic forward path for distributed modes."
            )
        if hidden_states.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("SonicGroupedMLP supports BF16/FP16 activations only.")
        if hidden_states.device.type != "cuda":
            raise ValueError("SonicGroupedMLP requires CUDA tensors.")
        if self.config.moe_apply_probs_on_input:
            raise ValueError(
                "SonicGroupedMLP routed forward does not support moe_apply_probs_on_input."
            )
        if not self._logged_routed_forward:
            logger.info(
                "SonicGroupedMLP using routed SonicMoE path; Megatron local expert "
                "permutation/combine is bypassed for this rank."
            )
            self._logged_routed_forward = True

        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, original_shape[-1])
        routing_map = routing_map.reshape(hidden_states.size(0), -1)
        probs = probs.reshape(hidden_states.size(0), -1)

        if routing_map.dtype != torch.bool:
            routing_map = routing_map.bool()
        if hidden_states.nelement() == 0 or routing_map.numel() == 0:
            output = hidden_states.new_empty(hidden_states.shape)
            zero = self.linear_fc1_weight.sum() * 0 + self.linear_fc2_weight.sum() * 0
            return (output + zero).view(original_shape), None

        try:
            from sonicmoe.enums import ActivationType as SonicActivationType
            from sonicmoe.functional import moe_general_routing_inputs
        except Exception as exc:
            raise ImportError(
                "SonicGroupedMLP requires the sonic-moe package. Install it in the "
                "training container, for example with `pip install 'sonic-moe[cu13]'`."
            ) from exc

        token_expert_indices = routing_map.nonzero(as_tuple=False)
        if token_expert_indices.numel() == 0:
            output = hidden_states.new_empty(hidden_states.shape)
            zero = self.linear_fc1_weight.sum() * 0 + self.linear_fc2_weight.sum() * 0
            return (output + zero).view(original_shape), None
        token_indices = token_expert_indices[:, 0].to(torch.int32)
        expert_indices = token_expert_indices[:, 1].to(torch.int32)
        router_scores = probs[routing_map]

        w1, w2 = self._sonic_weights()
        output, _ = moe_general_routing_inputs(
            hidden_states,
            router_scores,
            token_indices,
            expert_indices,
            w1,
            None,
            w2,
            None,
            self.num_local_experts,
            torch.cuda.current_stream().cuda_stream,
            SonicActivationType(self._sonic_activation_type()),
            not torch.is_grad_enabled(),
            False,
        )
        return output.view(original_shape), None

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for SonicMoE local experts."""
        if permuted_local_hidden_states.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("SonicGroupedMLP supports BF16/FP16 activations only.")
        if permuted_local_hidden_states.device.type != "cuda":
            raise ValueError("SonicGroupedMLP requires CUDA tensors.")

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            ).to(original_dtype)
            permuted_probs = torch.ones_like(permuted_probs)

        if permuted_local_hidden_states.nelement() == 0:
            return self._zero_token_forward(permuted_local_hidden_states, permuted_probs)

        w1, w2 = self._sonic_weights()
        try:
            from sonicmoe.enums import ActivationType as SonicActivationType
            from sonicmoe.functional import _DownProjection, _UpProjection
        except Exception as exc:
            raise ImportError(
                "SonicGroupedMLP requires the sonic-moe package. Install it in the "
                "training container, for example with `pip install 'sonic-moe[cu13]'`."
            ) from exc

        device = permuted_local_hidden_states.device
        num_tokens = permuted_local_hidden_states.size(0)
        tokens_per_expert = tokens_per_expert.to(device=device, dtype=torch.int32)
        real_probs = permuted_probs.view(num_tokens, 1)
        if real_probs.dtype != torch.float32:
            real_probs = real_probs.float()

        expert_frequency_offset = torch.empty(
            self.num_local_experts + 1, device=device, dtype=torch.int32
        )
        expert_frequency_offset[0].zero_()
        torch.cumsum(tokens_per_expert, dim=0, out=expert_frequency_offset[1:])

        identity_indices = torch.arange(num_tokens, device=device, dtype=torch.int32)
        token_offsets = torch.arange(num_tokens + 1, device=device, dtype=torch.int32)
        activation_type = SonicActivationType(self._sonic_activation_type())

        a, h = _UpProjection.apply(
            permuted_local_hidden_states,
            w1,
            None,
            expert_frequency_offset,
            num_tokens,
            1,
            identity_indices,
            identity_indices,
            identity_indices,
            token_offsets,
            False,
            activation_type,
            not torch.is_grad_enabled(),
            False,
        )

        output = _DownProjection.apply(
            a,
            h,
            w2,
            None,
            real_probs,
            expert_frequency_offset,
            num_tokens,
            1,
            identity_indices,
            identity_indices,
            identity_indices,
            token_offsets,
            False,
            activation_type,
        )
        return output, None

    def backward_dw(self):
        """Compatibility hook for overlapped MoE schedules."""
        pass

    def _merge_legacy_sonic_state_dict_keys(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        del local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        fc1_key = f"{prefix}linear_fc1_weight"
        fc2_key = f"{prefix}linear_fc2_weight"
        if fc1_key not in state_dict:
            legacy_fc1_keys = [
                f"{prefix}linear_fc1.weight{i}" for i in range(self.num_local_experts)
            ]
            if all(k in state_dict for k in legacy_fc1_keys):
                legacy_fc1_weight = torch.stack(
                    [state_dict.pop(k).transpose(0, 1) for k in legacy_fc1_keys], dim=0
                ).contiguous()
                state_dict[fc1_key] = self._interleave_glu_weight(legacy_fc1_weight)
        if fc2_key not in state_dict:
            legacy_fc2_keys = [
                f"{prefix}linear_fc2.weight{i}" for i in range(self.num_local_experts)
            ]
            if all(k in state_dict for k in legacy_fc2_keys):
                state_dict[fc2_key] = torch.stack(
                    [state_dict.pop(k).transpose(0, 1) for k in legacy_fc2_keys], dim=0
                ).contiguous()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        for local_idx in range(self.num_local_experts):
            fc1_weight = self._deinterleave_glu_weight(
                self.linear_fc1_weight[local_idx]
            ).transpose(0, 1)
            fc2_weight = self.linear_fc2_weight[local_idx].transpose(0, 1)
            if not keep_vars:
                fc1_weight = fc1_weight.detach()
                fc2_weight = fc2_weight.detach()
            destination[f"{prefix}linear_fc1.weight{local_idx}"] = fc1_weight
            destination[f"{prefix}linear_fc2.weight{local_idx}"] = fc2_weight

    def _packed_expert_weight_factory(
        self,
        base_key: str,
        weight: torch.nn.Parameter,
        per_expert_prefix: str,
        tp_axis: int,
        expert_axis: int,
        transpose_expert_weight: bool,
        split_glu: bool,
        sharded_offsets: tuple,
        metadata: Optional[dict],
    ) -> ShardedTensorFactory:
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        num_global_experts = self.ep_group.size() * self.num_local_experts
        local_expert_indices_offset = self.ep_group.rank() * self.num_local_experts
        ep_axis = len(sharded_offsets)

        def build_fn(key: str, tensor: torch.Tensor, replica_id, flattened_range):
            del key, replica_id
            if flattened_range is not None:
                if tensor.numel() != weight.numel():
                    raise ValueError(
                        "SonicGroupedMLP packed expert factory received a flat optimizer "
                        "range that does not cover the full packed parameter."
                    )
                tensor = tensor.view_as(weight)
            sharded_tensors = {}
            for local_idx in range(self.num_local_experts):
                global_idx = local_expert_indices_offset + local_idx
                expert_weight = tensor.select(expert_axis, local_idx)
                if split_glu:
                    expert_weight = self._deinterleave_glu_weight(expert_weight)
                if transpose_expert_weight:
                    expert_weight = expert_weight.transpose(0, 1)
                expert_key = f"{per_expert_prefix}weight{local_idx}"
                expert_offsets = (
                    *sharded_offsets,
                    (ep_axis, global_idx, num_global_experts),
                )
                sharded_tensor = make_tp_sharded_tensor_for_checkpoint(
                    expert_weight,
                    expert_key,
                    tp_axis=tp_axis,
                    prepend_offsets=expert_offsets,
                    tp_group=self.tp_group,
                    dp_cp_group=metadata["dp_cp_group"],
                )
                if split_glu:
                    sharded_tensor = apply_swiglu_sharded_factory(
                        sharded_tensor, expert_offsets
                    ).build()
                sharded_tensors[f"weight{local_idx}"] = sharded_tensor
            return sharded_tensors

        def merge_fn(sub_state_dict):
            expert_weights = []
            for local_idx in range(self.num_local_experts):
                expert_weight = sub_state_dict[f"weight{local_idx}"]
                if isinstance(expert_weight, (list, tuple)):
                    expert_weight = torch.cat(expert_weight, dim=0)
                if transpose_expert_weight:
                    expert_weight = expert_weight.transpose(0, 1)
                if split_glu:
                    expert_weight = self._interleave_glu_weight(expert_weight)
                expert_weights.append(expert_weight)
            return torch.stack(expert_weights, dim=expert_axis).contiguous()

        replica_id = (0, self.tp_group.rank(), self.ep_group.rank())
        return ShardedTensorFactory(base_key, weight, build_fn, merge_fn, replica_id=replica_id)

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Expose packed Sonic runtime weights as legacy per-expert grouped weights."""
        return {
            f"{prefix}linear_fc1_weight": self._packed_expert_weight_factory(
                f"{prefix}linear_fc1_weight",
                self.linear_fc1_weight,
                f"{prefix}linear_fc1.",
                tp_axis=0,
                expert_axis=0,
                transpose_expert_weight=True,
                split_glu=True,
                sharded_offsets=sharded_offsets,
                metadata=metadata,
            ),
            f"{prefix}linear_fc2_weight": self._packed_expert_weight_factory(
                f"{prefix}linear_fc2_weight",
                self.linear_fc2_weight,
                f"{prefix}linear_fc2.",
                tp_axis=1,
                expert_axis=0,
                transpose_expert_weight=True,
                split_glu=False,
                sharded_offsets=sharded_offsets,
                metadata=metadata,
            ),
        }


class InferenceGroupedMLP(TEGroupedMLP):
    """Inference-optimized GroupedMLP with GPU-resident offsets.

    Inherits from TEGroupedMLP to reuse weight initialization and checkpoint compatibility.
    Supports three forward paths:
    - Training: delegates to parent TEGroupedMLP
    - Inference + CUDA graphed: FlashInfer cutlass_fused_moe (fused permute + GEMM)
    - Inference + eager: torch.nn.functional.grouped_mm with GPU-resident cumsum offsets
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        submodules: GroupedMLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        # Initialize parent TEGroupedMLP (creates linear_fc1, linear_fc2)
        super().__init__(
            num_local_experts=num_local_experts,
            config=config,
            submodules=submodules,
            pg_collection=pg_collection,
        )

        # Concatenated weights are built lazily on first forward to ensure
        # checkpoint loading has already populated the per-expert parameters.
        self._concatenated_weights_built = False

        self.is_inference_cuda_graphed_iteration = False

        if HAVE_FLASHINFER:
            self._flashinfer_activation_type = self._resolve_flashinfer_activation_type()

        self._mcore_activation_type = self._resolve_mcore_activation_type()
        self.inference_grouped_gemm_backend = config.inference_grouped_gemm_backend

    def _resolve_flashinfer_activation_type(self):
        """Map megatron activation config to FlashInfer ActivationType."""
        assert (
            HAVE_FLASHINFER
        ), "flashinfer-python is required to resolve FlashInfer activation type."
        func = self.config.activation_func
        if func == F.silu:
            return ActivationType.Silu
        elif func == F.gelu:
            return ActivationType.Gelu
        elif func == F.relu:
            return ActivationType.Relu
        elif func == squared_relu:
            return ActivationType.Relu2
        raise ValueError(f"No FlashInfer ActivationType mapping for activation_func={func}")

    def _resolve_mcore_activation_type(self):
        """Map megatron activation config to mcore_fused_moe ActivationType."""
        func = self.config.activation_func
        if func == squared_relu:
            return McoreActivationType.SQUARED_RELU
        raise ValueError(f"No mcore_fused_moe ActivationType mapping for activation_func={func}")

    def set_inference_cuda_graphed_iteration(self):
        """Enable CUDA-graphed iteration mode."""
        self.is_inference_cuda_graphed_iteration = True

    def unset_inference_cuda_graphed_iteration(self):
        """Disable CUDA-graphed iteration mode."""
        self.is_inference_cuda_graphed_iteration = False

    def _build_concatenated_mxfp8_weights(self):
        """Build stacked MXFP8 weight tensors from per-expert MXFP8Tensor attributes.

        After quantize_model_to_mxfp8, each per-expert weight (weight0, weight1, ...)
        has been replaced with an MXFP8Tensor. This method stacks their data and
        scales into _fc1_weight / _fc2_weight for scaled_grouped_mm.

        Note: this creates a contiguous copy since per-expert MXFP8Tensor attributes
        are not contiguous across experts. This is a one-time cost at first forward.

        Unlike _build_concatenated_weights, this does not create nn.Parameter views
        back into the buffer — MXFP8 weights are not nn.Parameters (they are plain
        MXFP8Tensor attributes set by quantize_model_to_mxfp8). This path is only
        intended for non-colocated inference.
        """

        for linear_name, buf_name in [('linear_fc1', '_fc1_weight'), ('linear_fc2', '_fc2_weight')]:
            linear = getattr(self, linear_name)
            q_list, s_list = [], []
            for i in range(self.num_local_experts):
                w = getattr(linear, f'weight{i}')
                if isinstance(w, MXFP8Tensor):
                    mxfp8 = w
                elif hasattr(w, 'data') and isinstance(w.data, MXFP8Tensor):
                    mxfp8 = w.data
                else:
                    raise RuntimeError(
                        f"Expected MXFP8Tensor for {linear_name}.weight{i}, "
                        f"got {type(w).__name__}. Was quantize_model_to_mxfp8 called?"
                    )
                q_list.append(mxfp8.data)
                s_list.append(mxfp8.scale)

            stacked_data = torch.stack(q_list, dim=0).contiguous()
            stacked_scale = torch.stack(s_list, dim=0).contiguous()

            setattr(self, buf_name, MXFP8Tensor(data=stacked_data, scale=stacked_scale))

            # Redirect per-expert weight .data to views into the stacked buffer,
            # mirroring _build_concatenated_weights. This frees the original
            # allocations while keeping the Parameter objects intact.
            for i in range(self.num_local_experts):
                w = getattr(linear, f'weight{i}')
                if isinstance(w, MXFP8Tensor):
                    w.data = stacked_data[i]
                    w.scale = stacked_scale[i]
                elif hasattr(w, 'data') and isinstance(w.data, MXFP8Tensor):
                    w.data.data = stacked_data[i]
                    w.data.scale = stacked_scale[i]

    @torch.inference_mode(False)  # needed for non-colocated inference.
    def _build_concatenated_weights(self):
        """Create big contiguous weight tensors that share storage with TE's per-expert parameters.

        Creates _fc1_weight and _fc2_weight as contiguous tensors of shape
        [num_experts, out_features, in_features]. Instead of replacing TE's parameters
        (which breaks TE's internal bookkeeping), we redirect each parameter's .data
        to be a view into the contiguous buffer. The nn.Parameter objects themselves
        remain untouched in TE's module, preserving FP8 scaling state, etc.

        This allows:
        - TE's forward to work correctly (same Parameter objects, same internal state)
        - Training updates to flow through (param.data is a view into the big tensor)
        - torch.nn.functional.grouped_mm / FlashInfer to use the big tensor directly
        """
        # Get device/dtype from existing TE weights
        device = self.linear_fc1.weight0.device
        dtype = self.linear_fc1.weight0.dtype

        fc1_shape = self.linear_fc1.weight0.shape  # [out_features, in_features]
        fc2_shape = self.linear_fc2.weight0.shape

        # Create big contiguous tensors
        _fc1_weight = torch.empty(self.num_local_experts, *fc1_shape, device=device, dtype=dtype)
        _fc2_weight = torch.empty(self.num_local_experts, *fc2_shape, device=device, dtype=dtype)

        # Copy existing TE weights into big tensors, then point param.data to the views
        for i in range(self.num_local_experts):
            fc1_param = getattr(self.linear_fc1, f'weight{i}')
            fc2_param = getattr(self.linear_fc2, f'weight{i}')

            # Copy initialized data into contiguous buffer
            _fc1_weight[i].copy_(fc1_param.data)
            _fc2_weight[i].copy_(fc2_param.data)

            # Redirect param.data to view into contiguous buffer.
            # The nn.Parameter object stays the same — TE's internal state is preserved.
            fc1_param.data = _fc1_weight[i]
            fc2_param.data = _fc2_weight[i]

        # Register big tensors as non-persistent buffers (for .to() device movement, not saved)
        self.register_buffer('_fc1_weight', _fc1_weight, persistent=False)
        self.register_buffer('_fc2_weight', _fc2_weight, persistent=False)

    def _flashinfer_forward(self, hidden_states, routing_map, probs):
        """FlashInfer fused MoE kernel for CUDA-graphed inference iterations."""
        assert HAVE_FLASHINFER, "flashinfer-python is required for FlashInfer forward path."
        assert probs.dtype == torch.float32, "FlashInfer forward path requires fp32 probabilities."
        output = fused_moe.cutlass_fused_moe(
            hidden_states,
            routing_map.int(),
            probs,
            self._fc1_weight,
            self._fc2_weight,
            hidden_states.dtype,
            quant_scales=None,
            activation_type=self._flashinfer_activation_type,
            ep_size=self.ep_group.size(),
            ep_rank=self.ep_group.rank(),
        )[0]
        return output, None

    def _mcore_fused_moe_forward(
        self, hidden_states, probs, routing_map=None, tokens_per_expert=None, skip_permute=False
    ):
        """Torch grouped_mm fused MoE forward via mcore_fused_moe."""
        local_expert_start = self.ep_group.rank() * self.num_local_experts
        output = mcore_fused_moe(
            hidden_states,
            probs,
            self._fc1_weight,
            self._fc2_weight,
            activation_type=self._mcore_activation_type,
            num_local_experts=self.num_local_experts,
            local_expert_start=local_expert_start,
            routing_map=routing_map,
            tokens_per_expert=tokens_per_expert,
            skip_permute=skip_permute,
            disable_fused_quant_kernels=self.config.inference_moe_disable_fused_quant_kernels,
        )
        return output, None

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: Optional[torch.Tensor],
        permuted_probs: torch.Tensor,
        routing_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with three modes:

        - Training: delegates to parent TEGroupedMLP.
        - Inference + CUDA graphed: FlashInfer cutlass_fused_moe. tokens_per_expert
          is not used in this path; the FlashInfer kernel operates directly on
          routing_map.
        - Inference + eager: torch.nn.functional.grouped_mm with GPU-resident cumsum offsets.

        Args:
            permuted_local_hidden_states: [num_tokens, hidden_size] input hidden states.
            tokens_per_expert: [num_experts] number of tokens routed to each expert.
                None when using the CUDA-graphed FlashInfer path.
            permuted_probs: [num_tokens, topk] routing probabilities.
            routing_map: [num_tokens, topk] token-to-expert assignment indices.
                Required for the FlashInfer CUDA-graphed path, None otherwise.
        """

        if self.training:
            assert (
                not self.config.fp8_recipe == "mxfp8"
            ), "MXFP8 inference optimized is not compatible with training / colocated RL."
            return super().forward(permuted_local_hidden_states, tokens_per_expert, permuted_probs)

        # Lazily build concatenated weights on first forward (after checkpoint load)
        if not self._concatenated_weights_built:
            w = self.linear_fc1.weight0
            if isinstance(w, MXFP8Tensor) or (
                hasattr(w, 'data') and isinstance(w.data, MXFP8Tensor)
            ):
                self._build_concatenated_mxfp8_weights()
            else:
                self._build_concatenated_weights()
            self._concatenated_weights_built = True

        resolved_backend = resolve_inference_grouped_gemm_backend(
            self.inference_grouped_gemm_backend,
            self.is_inference_cuda_graphed_iteration,
            is_mxfp8=self.config.fp8_recipe == "mxfp8",
        )

        if resolved_backend == InferenceGroupedGemmBackend.FLASHINFER:
            assert routing_map is not None, "routing_map is required for FlashInfer forward pass."
            assert (
                self.is_inference_cuda_graphed_iteration
            ), "FlashInfer forward path is only used in CUDA-graphed inference iterations."
            return self._flashinfer_forward(
                permuted_local_hidden_states, routing_map, permuted_probs
            )
        elif resolved_backend == InferenceGroupedGemmBackend.TORCH:
            return self._mcore_fused_moe_forward(
                permuted_local_hidden_states,
                permuted_probs,
                routing_map=routing_map,
                tokens_per_expert=tokens_per_expert,
                skip_permute=(not self.is_inference_cuda_graphed_iteration),
            )
        elif resolved_backend == InferenceGroupedGemmBackend.TE:
            return super().forward(permuted_local_hidden_states, tokens_per_expert, permuted_probs)


class SequentialMLP(MegatronModule):
    """An implementation of the Experts layer using a sequence of MLP layers.

    This class executes each expert sequentially.
    """

    # TODO(M4): breaking api, switched from pass in tp_group to pass in pg_collection.
    def __init__(
        self,
        num_local_experts,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):

        if config.moe_ffn_hidden_size == config.ffn_hidden_size:
            super().__init__(config=config)
        else:
            # Local SequentialMLP can still be used here by overriding the ffn_hidden_size
            # with a deepcopied config.
            sequential_mlp_config = deepcopy(config)
            sequential_mlp_config.ffn_hidden_size = config.moe_ffn_hidden_size
            super().__init__(config=sequential_mlp_config)

        self.num_local_experts = num_local_experts
        self.local_experts = torch.nn.ModuleList()
        self.ep_group = pg_collection.ep
        self.tp_group = pg_collection.expt_tp
        # use pg_collection.expt_dp_group as data parallel group in this module.
        # TODO (Hepteract): expt_dp wont be needed here once distributed checkpoint is refactored
        self.dp_group = pg_collection.expt_dp

        for _ in range(self.num_local_experts):
            expert = MLP(
                self.config,
                submodules,
                ffn_hidden_size=self.config.moe_ffn_hidden_size,
                is_expert=True,
                tp_group=pg_collection.expt_tp,
            )
            self.local_experts.append(expert)

    def _pad_tensor_for_quantization(self, hidden, probs):
        """Padding tensor shape to multiples of 16/32."""
        actual_num_tokens = hidden.shape[0]
        divisor = get_align_size_for_quantization(self.config)
        padded_num_tokens = ceil(actual_num_tokens / divisor) * divisor - actual_num_tokens
        if padded_num_tokens > 0:
            pad_tensor = torch.zeros(
                padded_num_tokens, hidden.shape[1], dtype=hidden.dtype, device=hidden.device
            )
            hidden = torch.cat((hidden, pad_tensor), dim=0)
            pad_probs = torch.zeros(padded_num_tokens, dtype=probs.dtype, device=probs.device)
            probs = torch.cat((probs, pad_probs), dim=0)
        return hidden, probs

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """Forward step of the SequentialMLP."""

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        if self.num_local_experts == 1:
            if self.config.fp8 or self.config.fp4:
                hidden, probs = self._pad_tensor_for_quantization(
                    permuted_local_hidden_states, permuted_probs
                )
                output, output_bias = self.local_experts[0](hidden, probs)
                output = output[: permuted_local_hidden_states.shape[0]]
            else:
                output, output_bias = self.local_experts[0](
                    permuted_local_hidden_states, permuted_probs
                )

            return output, output_bias
        else:
            tokens_per_expert = tokens_per_expert.tolist()
            tokens_list = torch.split(permuted_local_hidden_states, tokens_per_expert)
            probs_list = torch.split(permuted_probs, tokens_per_expert)

            output_local_list = []

            for expert, tokens, probs in zip(self.local_experts, tokens_list, probs_list):
                if self.config.fp8 or self.config.fp4:
                    hidden, probs = self._pad_tensor_for_quantization(tokens, probs)
                    output, output_bias = expert(hidden, probs)
                    output = output[: tokens.shape[0]]
                else:
                    output, output_bias = expert(tokens, probs)
                output_local_list.append(output)

            output_local = torch.cat(output_local_list, dim=0)
            output_bias_local = None
            # Note: if bias is enabled on experts, it is already added to the output at this point
            return output_local, output_bias_local

    def backward_dw(self):
        """Backward pass for weight gradients in SequentialMLP."""
        for expert in self.local_experts:
            expert.backward_dw()

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Maps local expert to global experts."""
        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)

        sharded_state_dict = {}
        num_global_experts = self.ep_group.size() * self.num_local_experts
        local_expert_indices_offset = self.ep_group.rank() * self.num_local_experts

        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)

        for expert_local_idx, expert in enumerate(self.local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            expert_state_dict_prefix = f'{prefix}local_experts.{expert_local_idx}.'
            if singleton_local_shards:
                expert_sharded_prefix = f'{prefix}experts.{expert_global_idx}.'
                expert_sharded_offsets = sharded_offsets
            else:
                expert_sharded_prefix = f'{prefix}experts.'
                expert_sharded_offsets = (
                    *sharded_offsets,
                    (len(sharded_offsets), expert_global_idx, num_global_experts),
                )

            expert_state_dict = expert.sharded_state_dict(
                expert_state_dict_prefix, expert_sharded_offsets, metadata
            )
            # Remove expert layers indexing from sharded keys
            replace_prefix_for_sharding(
                expert_state_dict, expert_state_dict_prefix, expert_sharded_prefix
            )
            # Adjust replica ids - replication along DP modulo EP
            for k, sh_ten in expert_state_dict.items():
                replica_id = sh_ten.replica_id
                assert (
                    len(replica_id) == 3
                ), f'Expected replica_id for {k} to be in (PP, TP, DP) format, got: {replica_id}'

                sh_ten.replica_id = (*replica_id[:2], self.dp_group.rank())

            sharded_state_dict.update(expert_state_dict)
        return sharded_state_dict
