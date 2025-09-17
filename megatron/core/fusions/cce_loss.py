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
import torch.nn.functional as F

from megatron.core.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
)
from megatron.core.tensor_parallel import (
        reduce_from_tensor_model_parallel_region
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

def generate_fake_teacher_data(
    batch_size: int, 
    seq_length: int, 
    vocab_size: int, 
    num_teacher_tokens_per_pos: int = 3,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
) -> list:
    """Generate fake teacher data for testing distillation.
    
    Args:
        batch_size: Number of batch elements
        seq_length: Sequence length
        vocab_size: Vocabulary size
        num_teacher_positions: Number of positions per batch with teacher data
        num_teacher_tokens_per_pos: Number of teacher tokens per position
        device: Device for tensors
        dtype: Data type for values
        
    Returns:
        List of teacher data dictionaries, one per batch element
    """
    if device is None:
        device = torch.device('cpu')
        
    teacher_data = []
    
    for batch_idx in range(batch_size):
        # Random positions in the sequence
        positions = torch.arange(0, seq_length, device=device)
        
        batch_teacher = {
            'positions': [],
            'indices': [],  
            'values': []
        }
        
        for pos in positions:
            # Random vocabulary indices for this position
            teacher_indices, _ = torch.sort(torch.randint(0, vocab_size, (num_teacher_tokens_per_pos,), device=device))
            
            # Random probability values (will be normalized to sum to 1)
            teacher_values = torch.rand(num_teacher_tokens_per_pos, device=device, dtype=dtype)
            
            batch_teacher['positions'].append(pos)
            batch_teacher['indices'].append(teacher_indices)
            batch_teacher['values'].append(teacher_values)
            
        teacher_data.append(batch_teacher)
    
    return teacher_data

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
    teacher_data: Optional[list] = None, # List of dicts with 'positions', 'indices', 'values' per batch element
) -> torch.Tensor:
    """Compute (optionally shifted) per-token cross-entropy via CCE.
    Returns [B, T-1] if shift=True, else [B, T].
    """
    embeddings = embeddings.transpose(0, 1).contiguous()  # -> [B, S, H]
    if embeddings.size(0) != labels.size(0) or embeddings.size(1) != labels.size(1):
        raise RuntimeError(
            f"CCE shape mismatch after normalization: embeddings[...,:] has {embeddings.size()[:-1]} "
            f"but labels_local have {labels.size()} (expected embeddings.size()[:-1] == labels_local.size())."
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
    losses,lse = linear_cross_entropy(
        embeddings,                      # [B, T, H]
        classifier_weight,               # [V/TP_SIZE, H]
        labels,                    # [B, T]
        impl=impl,
        reduction=reduction,
        shift=int(shift),
        ignore_index=ignore_index,
        vocab_parallel_options=vp_opts,
        return_lse=True,
    )

    tp_world = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    tensor_parallel = tp_world > 1

    print("embeddings:", embeddings.size(), "classifier_weight:", classifier_weight.size(), "labels:", labels.size())
    # Generate fake teacher data if not provided (for testing)
    if teacher_data is None:
        print("No teacher data provided, generating fake teacher data for testing...")
        # Infer batch size and sequence length from embeddings and labels
        if embeddings.ndim == 3 and labels is not None:
            if embeddings.size(0) == labels.size(0):  # [B, T, H] format
                batch_size, seq_length = labels.size(0), labels.size(1)
            elif embeddings.size(1) == labels.size(0):  # [T, B, H] format  
                batch_size, seq_length = labels.size(0), labels.size(1)
            else:
                batch_size, seq_length = labels.size(0), labels.size(1)
        else:
            raise ValueError("Cannot infer batch_size and seq_length from embeddings/labels")
            
        teacher_data = generate_fake_teacher_data(
            batch_size=batch_size,
            seq_length=seq_length, 
            vocab_size=vocab_size,
            num_teacher_tokens_per_pos=10,
            device=embeddings.device,
            dtype=embeddings.dtype
        )
   
    #print(embeddings.size(), classifier_weight.size(), labels.size(), losses.size(), lse.size())
    dev = embeddings.device
    dth = embeddings.dtype
    B, Tlen, H = embeddings.shape
    total_kl_sum = torch.zeros((), device=dev, dtype=dth)
    start, end = VocabUtility.vocab_range_from_global_vocab_size(vocab_size, tp_rank, tp_world)
    total_kl_cnt = 0

    # Process each batch element 
    for batch_idx in range(B):
        # Build teacher lookup for this batch element
        teacher_by_pos = {}
        batch_teacher_data = teacher_data[batch_idx]
        print(f"start : {start}, end : {end} for tp_rank {tp_rank} / {tp_world}")
        for p, idxs, vals in zip(batch_teacher_data.get('positions', []),
                                 batch_teacher_data.get('indices',  []),
                                 batch_teacher_data.get('values',   [])):
            pos = int(p.item() if isinstance(p, torch.Tensor) else p)
                
            ti = torch.as_tensor(idxs, device=dev, dtype=torch.long).flatten()
            tv = torch.as_tensor(vals, device=dev, dtype=dth).flatten()
            teacher_by_pos[pos] = (ti, tv)

        labels_1d = labels[batch_idx]          # [T]
        hidden_1d = embeddings[batch_idx]  # [T, H]
        logZ_full_T = lse[batch_idx].view(-1)  # [T]

        batch_kl_sum = torch.zeros((), device=dev, dtype=dth)
        batch_kl_cnt = 0
        chunk_size = 4096
        T = 1

        for t0 in range(0, Tlen, chunk_size):
            t1 = min(t0 + chunk_size, Tlen)
            
            # Get all positions in chunk that have teacher data
            chunk_positions = []
            chunk_teacher_indices = []
            chunk_teacher_values = []
            chunk_pos_in_chunk = []  # position within the chunk [0, chunk_length)
            
            for pos in range(t0, t1):
                if pos in teacher_by_pos and labels_1d[pos].item() != ignore_index:
                    t_idx, t_val = teacher_by_pos[pos]

                    t_logK = F.log_softmax(t_val / T, dim=0, dtype=torch.float32)  # [K]  # teacher log-probs at temp T

                    # Now filter for local vocabulary partition
                    if tensor_parallel:
                        # Mask for indices that belong to this TP rank's vocabulary partition
                        local_mask = (t_idx >= start) & (t_idx < end)
                        if not local_mask.any():
                            continue  # No relevant indices for this TP rank
                        t_idx_local = t_idx[local_mask] - start  # Convert to local indices
                        t_logK_local = t_logK[local_mask]
                    else:
                        t_idx_local = t_idx
                        t_logK_local = t_logK
                        local_mask = torch.ones_like(t_idx, dtype=torch.bool)
                    
                    chunk_positions.append(pos)
                    chunk_teacher_indices.append(t_idx_local)
                    chunk_teacher_values.append(t_logK_local)
                    chunk_pos_in_chunk.append(pos - t0)
            
            if not chunk_positions:
                continue
                
            # Convert to tensors for vectorized operations
            chunk_pos_in_chunk = torch.tensor(chunk_pos_in_chunk, device=dev, dtype=torch.long)
            
            # Pad teacher indices/values to same length for batching
            max_k = max(len(idx) for idx in chunk_teacher_indices)
            
            # Stack teacher data - pad shorter sequences
            teacher_indices_batch = torch.full((len(chunk_positions), max_k), -1, 
                                             device=dev, dtype=torch.long)
            teacher_values_batch = torch.zeros((len(chunk_positions), max_k),
                                             device=dev, dtype=dth)
            teacher_mask = torch.zeros((len(chunk_positions), max_k), 
                                     device=dev, dtype=torch.bool)

            for i, (t_idx, t_logK) in enumerate(zip(chunk_teacher_indices, chunk_teacher_values)):
                k = len(t_idx)
                teacher_indices_batch[i, :k] = t_idx
                teacher_values_batch[i, :k] = t_logK
                teacher_mask[i, :k] = True
            
            # Get unique teacher indices for this chunk (for computing union)
            valid_teacher_indices = teacher_indices_batch[teacher_mask]
            
            if len(valid_teacher_indices) == 0:
                continue
                
            # Union: teacher indices in chunk
            U_ids = torch.unique(valid_teacher_indices, sorted=True)
            
            # Compute logits for union 
            W_U = classifier_weight.index_select(0, U_ids)            # [U_local, H]
            x = hidden_1d[t0:t1]                                      # [L, H]
            #logits_chunk_U  = x @ W_U.t().contiguous()
            logits_chunk_U = F.linear(x, W_U, bias=None)

            # KL COMPUTATION for all positions in chunk
            # Map teacher indices to positions in U_ids
            teacher_cols = torch.searchsorted(U_ids, teacher_indices_batch)  # [num_positions, max_k]

            # Get student logits for each position's teacher indices
            pos_logits = logits_chunk_U[chunk_pos_in_chunk]  # [num_positions, U]
            
            # Gather teacher columns for each position 
            gathered_logits = torch.gather(pos_logits, 1, teacher_cols)  # [num_positions, max_k]
            
            # Apply temperature and compute student log-probs 
            s_scaled = gathered_logits / T  # [num_positions, max_k]
            logZ_positions = logZ_full_T[chunk_positions].unsqueeze(1)  # [num_positions, 1]
            s_logpK = s_scaled - logZ_positions  # [num_positions, max_k]
            
            # Compute KL divergence using traditional manual method
            # KL(teacher || student) = sum(teacher_prob * (teacher_logprob - student_logprob))
            # Only sum over valid (non-masked) elements
            kl_per_pos_per_k = teacher_values_batch.exp() * (teacher_values_batch - s_logpK)
            kl_per_pos_per_k = kl_per_pos_per_k * teacher_mask.float()  # zero out invalid positions
            kl_per_pos = kl_per_pos_per_k.sum(dim=1)  # [num_positions]
            if 5123 in chunk_positions:
                index = chunk_positions.index(5123)
                print(f"batch {batch_idx} efficient pos 5123 teacher_values_batch: {teacher_values_batch[index].exp()}")
                print(f"batch {batch_idx} efficient pos 6000 s_logpK: {s_logpK[index] * teacher_mask[index].float()} with logZ {logZ_positions[index]}")
                print(f"batch {batch_idx} efficient pos 6000 kl_per_pos: {kl_per_pos[index]}")
                print(f"batch {batch_idx} size of kl_per_pos: {kl_per_pos.size()}")
                
            
            chunk_kl_sum = kl_per_pos.sum()
            batch_kl_sum += chunk_kl_sum
            batch_kl_cnt += len(valid_teacher_indices)

        total_kl_sum += batch_kl_sum
        total_kl_cnt += batch_kl_cnt

    # Final computation with parallelism handling
    print(f"KL loss before TP/SP reduction: total_cnt: {total_kl_cnt}, total_sum: {total_kl_sum}")
    # For tensor parallelism, we need to reduce KL loss across TP ranks
    # since each rank computed loss only for its vocabulary partition
    if tensor_parallel:
        # Gather counts and losses from all TP ranks
        kl_sum_gathered = reduce_from_tensor_model_parallel_region(total_kl_sum) 
        kl_cnt_tensor = torch.tensor(total_kl_cnt, device=dev, dtype=torch.long)
        kl_cnt_gathered = reduce_from_tensor_model_parallel_region(kl_cnt_tensor)
        
        if kl_cnt_gathered > 0:
            kl_loss = (kl_sum_gathered / (kl_cnt_gathered /10)) * (T * T)
        else:
            kl_loss = torch.zeros_like(kl_sum_gathered)
            
        print(f"KL loss after SP reduction: {kl_loss}, total_cnt: {kl_cnt_gathered}")

    return losses, teacher_data