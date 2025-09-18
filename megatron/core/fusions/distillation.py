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
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

from megatron.core.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
from megatron.core.tensor_parallel import (
        reduce_from_tensor_model_parallel_region
    )
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.core.fusions.cce_loss import cce_per_token_loss


def distillation_loss(
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
    temp: float = 1.0, # temperature (a.k.a. tau)
    chunk_size: int = 1024,             # process sequence in chunks to save memory
    debug: bool = False,

) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """Compute (optionally shifted) per-token cross-entropy via CCE.

    Returns a tuple ``(losses, kl_loss, teacher_data)`` where ``losses`` retains
    the per-token shape ([B, T-1] if ``shift`` else [B, T]) and ``kl_loss`` is a
    tensor of shape [B, T] with per-token distillation losses.
    """

    loss, lse = cce_per_token_loss(
        embeddings=embeddings,
        classifier_weight=classifier_weight,
        labels=labels,
        vocab_size=vocab_size,
        impl=impl,
        reduction=reduction,
        shift=int(shift),
        ignore_index=ignore_index,
        return_lse=True,
        temp=temp,
    )
    embeddings = embeddings.transpose(0, 1).contiguous()  # -> [B, S, H]

    tp_world = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    tensor_parallel = tp_world > 1
    start, end = VocabUtility.vocab_range_from_global_vocab_size(vocab_size, tp_rank, tp_world)
    dev = embeddings.device
    B, Tlen, H = embeddings.shape
    
    # Initialize per-token KL loss tensor
    kl_losses = torch.zeros((B, Tlen), device=dev, dtype=torch.float32)

    # Generate fake teacher data if not provided (for testing)
    if teacher_data is None:
        print("No teacher data provided, generating fake teacher data for testing...")
        # Infer batch size and sequence length from embeddings and labels
        seq_length = labels.size(1)
            
        teacher_data = generate_fake_teacher_data(
            batch_size=B,
            seq_length=seq_length,
            vocab_size=vocab_size,
            num_teacher_tokens_per_pos=10,
            device=embeddings.device,
            dtype=embeddings.dtype
        )
    elif teacher_data is None:
        raise ValueError("teacher_data must be provided for distillation loss computation")

    # Process each batch element 
    for batch_idx in range(B):
        # Build teacher lookup for this batch element
        teacher_by_pos = {}
        batch_teacher_data = teacher_data[batch_idx]
        for p, idxs, vals in zip(batch_teacher_data.get('positions', []),
                                 batch_teacher_data.get('indices',  []),
                                 batch_teacher_data.get('values',   [])):
            pos = int(p.item() if isinstance(p, torch.Tensor) else p)
                
            ti = torch.as_tensor(idxs, device=dev, dtype=torch.long).flatten()
            tv = torch.as_tensor(vals, device=dev).flatten()
            teacher_by_pos[pos] = (ti, tv)

        labels_1d = labels[batch_idx]          # [T]
        hidden_1d = embeddings[batch_idx]  # [T, H]
        logZ_full_T = lse[batch_idx].view(-1)  # [T]

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

                    t_logK = F.log_softmax(t_val / temp, dim=0, dtype=torch.float32)  # [K]  # teacher log-probs at temp T

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
                                             device=dev)
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
            logits_chunk_U  = x @ W_U.t().contiguous()

            # KL COMPUTATION for all positions in chunk
            # Map teacher indices to positions in U_ids
            teacher_cols = torch.searchsorted(U_ids, teacher_indices_batch)  # [num_positions, max_k]

            # Get student logits for each position's teacher indices
            pos_logits = logits_chunk_U[chunk_pos_in_chunk]  # [num_positions, U]
            
            # Gather teacher columns for each position 
            gathered_logits = torch.gather(pos_logits, 1, teacher_cols)  # [num_positions, max_k]
            
            # Apply temperature and compute student log-probs 
            s_scaled = gathered_logits / temp  # [num_positions, max_k]
            logZ_positions = logZ_full_T[chunk_positions].unsqueeze(1)  # [num_positions, 1]
            s_logpK = s_scaled - logZ_positions  # [num_positions, max_k]
            
            # Compute KL divergence using traditional manual method
            # KL(teacher || student) = sum(teacher_prob * (teacher_logprob - student_logprob))
            # Only sum over valid (non-masked) elements
            kl_per_pos_per_k = teacher_values_batch.exp() * (teacher_values_batch - s_logpK)
            kl_per_pos_per_k = kl_per_pos_per_k * teacher_mask.float()  # zero out invalid positions
            kl_per_pos = kl_per_pos_per_k.sum(dim=1)  # [num_positions]
            
            # Store per-position KL losses in the output tensor
            # Apply temperature scaling
            kl_per_pos_scaled = kl_per_pos * (temp ** 2)
            
            # Assign to the correct positions in the output tensor
            kl_losses[batch_idx, chunk_positions] = kl_per_pos_scaled

            if debug:
                index_to_check = 320
                if index_to_check >= t0 and index_to_check < t1:
                    index = chunk_positions.index(index_to_check)
                    print(f"batch {batch_idx} efficient pos {index_to_check} teacher_values_batch: {teacher_values_batch[index].exp()}")
                    print(f"batch {batch_idx} efficient pos {index_to_check} s_logpK: {s_logpK[index] * teacher_mask[index].float()} with logZ {logZ_positions[index]}")
                    print(f"batch {batch_idx} efficient pos {index_to_check} kl_per_pos: {kl_per_pos[index]}")

    # For tensor parallelism, we need to reduce KL loss across TP ranks
    # since each rank computed loss only for its vocabulary partition
    if tensor_parallel:
        # Gather losses from all TP ranks
        kl_losses = reduce_from_tensor_model_parallel_region(kl_losses)

    if debug:
        print(f"  Number of elements: {kl_losses.shape}")
        print(f"  KL losses: {kl_losses.sum()/(B*Tlen)}")
        return loss, kl_losses, teacher_data
    
    return loss, kl_losses


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


def traditional_distillation_loss(
    student_logits: torch.Tensor,      # (B, T, V) unnormalized logits
    teacher_data: list,                # List of teacher data dictionaries, one per batch element
    labels: torch.Tensor,              # (B, T) integer class ids, with ignore_index masked out
    T: float = 1.0,                    # temperature (a.k.a. tau)
    ignore_index: int = -100,
):
    """
    Knowledge Distillation (Hinton et al., 2015) using teacher logits.

    We compute KL(P_teacher || Q_student) where:
      - P_teacher is a softmax over the teacher-provided logits at temperature T
      - Q_student is the student's log-softmax over the FULL vocab at temperature T,
        gathered at the teacher indices (so normalization is still over V).
    The KD term is scaled by T**2 as recommended.

    Args:
        student_logits: (B, T, V) unnormalized logits from student model
        teacher_data: List of teacher data dictionaries, one per batch element.
                     Each dict has 'positions', 'indices', 'values' keys where:
                     - positions: list of sequence positions with teacher data
                     - indices: list of teacher token indices for each position
                     - values: list of teacher logit values for each position
        labels: (B, T) integer class ids, with ignore_index masked out
        T: temperature for softmax
        ignore_index: index to ignore in loss computation

    Returns:
        KL divergence loss value
    """
    student_logits=student_logits.transpose(0, 1).contiguous()
    device = student_logits.device
    B, Tseq, V = student_logits.shape
    
    if len(teacher_data) != B:
        raise ValueError(f"teacher_data length {len(teacher_data)} must match batch size {B}")

    total_kl_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_valid_positions = torch.zeros((), device=device, dtype=torch.float32)
    index_to_check = 320

    # Process each batch element
    for batch_idx in range(B):
        batch_teacher = teacher_data[batch_idx]
        batch_labels = labels[batch_idx]  # (T,)
        batch_student_logits = student_logits[batch_idx]  # (T, V)
        
        # Get teacher data for this batch element
        positions = batch_teacher.get('positions', [])
        indices_list = batch_teacher.get('indices', [])
        values_list = batch_teacher.get('values', [])
        
        if not positions or not indices_list or not values_list:
            continue  # No teacher data for this batch element
            
        batch_kl_sum = torch.zeros_like(total_kl_loss)
        valid_positions_in_batch = torch.zeros_like(total_valid_positions)
        
        # Process each position with teacher data
        for pos, teacher_indices, teacher_values in zip(positions, indices_list, values_list):
            pos_idx = int(pos.item() if isinstance(pos, torch.Tensor) else pos)
            
            # Skip if position is out of bounds or should be ignored
            if batch_labels[pos_idx].item() == ignore_index:
                continue
                
            # Convert teacher data to tensors
            teacher_indices = torch.as_tensor(teacher_indices, dtype=torch.long, device=device)
            teacher_values = torch.as_tensor(teacher_values, device=device)
                
            # Get student logits for this position
            student_logits_pos = batch_student_logits[pos_idx]  # (V,)
            
            # Student log-probabilities at temperature T (normalized over full vocab)
            student_log_probs_full = F.log_softmax(student_logits_pos / T, dim=-1, dtype=torch.float32)  # (V,)
            student_log_probs_teacher = student_log_probs_full[teacher_indices]  # (K,)
            
            # Teacher probabilities at temperature T (normalized over teacher indices only)
            teacher_probs = F.log_softmax(teacher_values / T, dim=-1, dtype=torch.float32)  # (K,)
            # KL divergence: KL(P_teacher || Q_student)
            # F.kl_div expects log-probs as input and probs as target
            kl_loss_pos = teacher_probs.exp() * (teacher_probs - student_log_probs_teacher)
            kl_loss_pos = kl_loss_pos.sum()  # scalar
            if pos==index_to_check:
                print(f"batch {batch_idx} traditional pos {index_to_check} teacher_probs: {teacher_probs.exp()}")
                print(f"batch {batch_idx} traditional pos {index_to_check} student_log_probs_teacher: {student_log_probs_teacher} lse student: {torch.logsumexp(student_logits_pos.float() / T, dim=-1)}")
                print(f"batch {batch_idx} traditional pos {index_to_check} kl loss {kl_loss_pos}")

            batch_kl_sum += kl_loss_pos
            valid_positions_in_batch += 1
            
        total_kl_loss += batch_kl_sum
        total_valid_positions += valid_positions_in_batch
        print(f"batch {batch_idx} traditional valid positions: {valid_positions_in_batch}, batch_kl_sum: {batch_kl_sum}")

    # Average over all valid positions and apply temperature scaling
    if total_valid_positions > 0:
        kd_loss = (total_kl_loss / total_valid_positions) * (T ** 2)
    else:
        kd_loss = torch.zeros_like(total_kl_loss)

    print(f"KD loss: {kd_loss}, valid positions: {total_valid_positions}")
    return kd_loss