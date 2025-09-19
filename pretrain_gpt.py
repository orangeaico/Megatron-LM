# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain and SFT GPT."""

import os
import time
import contextlib

import torch

from functools import partial
from typing import List, Optional, Tuple
from megatron.core import parallel_state
from megatron.training import inprocess_restart
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import StragglerDetector
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.teacher_data_utils import (
    has_teacher_data,
    unpack_teacher_batch,
)
from megatron.training.datasets.sft_dataset import SFTDataset
from model_provider import model_provider
from gpt_builders import gpt_builder

try:
    from megatron.post_training.arguments import add_modelopt_args, modelopt_args_enabled
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

# =========================
# Phase logger utilities
# =========================

PHASE_LOGGER = True
PHASE_LAYER_LOGGER = False

def _is_rank0():
    try:
        if parallel_state.is_unitialized() if hasattr(parallel_state, "is_unitialized") else False:
            return int(os.environ.get("RANK", "0")) == 0
        # Prefer Megatron’s notion of data-parallel rank if available
        if hasattr(parallel_state, "get_data_parallel_rank"):
            return parallel_state.get_data_parallel_rank() == 0
    except Exception:
        pass
    return int(os.environ.get("RANK", "0")) == 0

def _bytes(x: int) -> str:
    x = float(x)
    for u in ["B","KB","MB","GB","TB","PB"]:
        if x < 1024: return f"{x:.2f}{u}"
        x /= 1024
    return f"{x:.2f}EB"

def _mem_stats(device=None):
    if device is None:
        device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    alloc    = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_a   = torch.cuda.max_memory_allocated(device)
    peak_r   = torch.cuda.max_memory_reserved(device)
    return {
        "allocated": _bytes(alloc),
        "reserved":  _bytes(reserved),
        "peak_allocated": _bytes(peak_a),
        "peak_reserved":  _bytes(peak_r),
    }

def _barrier():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass

@contextlib.contextmanager
def mem_phase(name: str, do_barrier: bool = False):
    """Emit per-phase CUDA memory stats (allocated/reserved + peaks)."""
    if not PHASE_LOGGER or not torch.cuda.is_available():
        yield
        return
    device = torch.cuda.current_device()
    if do_barrier:
        _barrier()
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        dt = time.time() - t0
        stats = _mem_stats(device)
        if _is_rank0():
            print(
                f"[MEM][{name}] dt={dt:.3f}s | "
                f"alloc={stats['allocated']} res={stats['reserved']} "
                f"peak_alloc={stats['peak_allocated']} peak_res={stats['peak_reserved']}",
                flush=True,
            )

def attach_module_peaks(module: torch.nn.Module, device=None):
    """Optionally print per-leaf-module forward peaks on rank 0."""
    if not (PHASE_LOGGER and PHASE_LAYER_LOGGER and torch.cuda.is_available() and _is_rank0()):
        return
    if device is None:
        device = torch.cuda.current_device()

    for name, m in module.named_modules():
        # Skip container modules; focus on compute layers
        if any(True for _ in m.children()):
            continue

        def pre_hook(mod, inp, n=name):
            torch.cuda.reset_peak_memory_stats(device)

        def post_hook(mod, out, n=name):
            pa = torch.cuda.max_memory_allocated(device)
            pr = torch.cuda.max_memory_reserved(device)
            print(f"[LAYER][{n}] peak_alloc={_bytes(pa)} peak_res={_bytes(pr)}", flush=True)

        m.register_forward_pre_hook(pre_hook)
        m.register_forward_hook(post_hook)

# NVTX helpers (nice in Chrome/Nsight traces)
try:
    import torch.cuda.nvtx as nvtx
    def nvtx_range(name):
        return contextlib.ExitStack().__enter__() if not torch.cuda.is_available() else _NVTX(name)
    class _NVTX:
        def __init__(self, name): self.name=name
        def __enter__(self): 
            try: nvtx.range_push(self.name)
            except Exception: pass
        def __exit__(self, exc_type, exc, tb):
            try: nvtx.range_pop()
            except Exception: pass
except Exception:
    def nvtx_range(name):  # fallback no-op
        return contextlib.nullcontext()

stimer = StragglerDetector()


def get_batch(data_iterator):
    """Generate a batch."""
    # TODO: this is pretty hacky, find a better way
    if (not parallel_state.is_pipeline_first_stage(ignore_virtual=True)) and (
        not parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    ):
        return None, None, None, None, None, None

    # get batches based on the TP rank you are on
    with mem_phase("LOAD_BATCH", do_barrier=True), nvtx_range("LOAD_BATCH"):
        batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    teacher_packed = batch.pop('teacher_data', None)
    teacher_data = None
    if teacher_packed is not None and has_teacher_data(teacher_packed):
        teacher_data = unpack_teacher_batch(teacher_packed)

    tokens = batch['tokens']
    labels = batch['labels']
    loss_mask = batch['loss_mask']
    attention_mask = batch['attention_mask']
    position_ids = batch['position_ids']

    return tokens, labels, loss_mask, attention_mask, position_ids, teacher_data


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return loss_func_modelopt(loss_mask, output_tensor, model=model)

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {'lm loss': reporting_loss})


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            teacher_data,
        ) = get_batch(data_iterator)
    timers('batch-generator').stop()

    # Optional per-layer peaks (only once the model exists)
    if PHASE_LOGGER and PHASE_LAYER_LOGGER:
        # Attach hooks only once per process
        if not hasattr(model, "_phase_layer_hooks_attached"):
            attach_module_peaks(model)
            model._phase_layer_hooks_attached = True

    with stimer, mem_phase("FORWARD", do_barrier=True), nvtx_range("FORWARD"):
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            if return_schedule_plan:
                assert args.overlap_moe_expert_parallel_comm, \
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )
                return schedule_plan, partial(loss_func, loss_mask, model=model)
            else:
                output_tensor = model(
                    tokens,
                    position_ids,
                    attention_mask,
                    labels=labels,
                    loss_mask=loss_mask,
                    teacher_data=teacher_data,
                )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank():
    return (
        parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        or parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    ) and parallel_state.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        multiple_validation_sets=args.multiple_validation_sets,
        full_validation=args.full_validation,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.sft:
        dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_dataset_built_on_rank, config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


# Wrapper to attach per-layer hooks as soon as the model is created (works with pretrain())
def _model_provider_with_phase(gbuilder, *args, **kwargs):
    mdl = model_provider(gbuilder, *args, **kwargs)
    if PHASE_LOGGER and PHASE_LAYER_LOGGER:
        try:
            if not hasattr(mdl, "_phase_layer_hooks_attached"):
                attach_module_peaks(mdl)
                mdl._phase_layer_hooks_attached = True
        except Exception:
            pass
    return mdl


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
    train_valid_test_datasets_provider,
    partial(_model_provider_with_phase, gpt_builder),  # <-- wrapped to attach layer peaks
    ModelType.encoder_or_decoder,
    forward_step,
    args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
    store=store,
    )
