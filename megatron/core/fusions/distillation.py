# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0
"""Knowledge Distillation with Tensor Parallelism for Large Language Models.

This module provides efficient knowledge distillation functionality for large language models
with support for tensor parallelism and memory-efficient processing. It adapts Megatron-LM's
tensor-parallel vocabulary sharding to work with apple/ml-cross-entropy's cross-entropy
computation for scalable distillation training.

The module implements both traditional knowledge distillation and an efficient chunked approach
that processes sequences in memory-friendly chunks while maintaining mathematical equivalence.
All computations are compatible with Megatron-LM's distributed training paradigms.

Key Features:
    - Tensor-parallel vocabulary processing for large vocabularies
    - Memory-efficient chunked sequence processing
    - Support for both standard and temperature-scaled distillation
    - Automatic teacher data generation for testing
    - Debug utilities for monitoring loss computation

Dependencies:
    pip install "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"

References:
    - Hinton et al. (2015): "Distilling the Knowledge in a Neural Network"
    - Repository: https://github.com/apple/ml-cross-entropy

Example:
    Basic usage for knowledge distillation:
    
    >>> # Standard distillation loss computation
    >>> kl_loss, teacher_data = distillation_loss(
    ...     embeddings=student_embeddings,
    ...     classifier_weight=vocab_projection_weights,
    ...     labels=target_labels,
    ...     vocab_size=50000,
    ...     teacher_data=teacher_logits_data,
    ...     temp=3.0
    ... )
"""

from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn.functional as F

from megatron.core.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_context_parallel_world_size,
    get_context_parallel_rank,
)
from megatron.core.tensor_parallel import reduce_from_tensor_model_parallel_region
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.core.fusions.cce_loss import cce_per_token_loss

# Constants
DEFAULT_TEMPERATURE = 1.0
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_IGNORE_INDEX = -100
DEFAULT_NUM_TEACHER_TOKENS = 50


def distillation_loss(
    *,
    embeddings: torch.Tensor,
    classifier_weight: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    impl: str = "cce",
    reduction: str = "none", 
    shift: bool = True,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    teacher_data: Optional[List[Dict[str, Any]]] = None,
    temp: float = DEFAULT_TEMPERATURE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    debug: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """
    Compute knowledge distillation loss using Cross-Entropy with Efficient implementation.
    
    This function computes both the standard cross-entropy loss and KL divergence loss
    for knowledge distillation. It supports tensor parallelism for distributed training
    and processes sequences in chunks to manage memory usage.
    
    Args:
        embeddings: Input embeddings tensor of shape [B, T, H] or [T, B, H] or [B, T_local, H] with sequence parallelism
        classifier_weight: Weight matrix of shape [V, H] (or [V_local, H] with vocab parallelism)
        labels: Target labels of shape [B, T_global] (global with respect to sequence parallelism)
        vocab_size: Total vocabulary size
        impl: Implementation type for CCE loss computation (default: "cce")
        reduction: Reduction method for loss (default: "none")
        shift: Whether to shift labels for next-token prediction (default: True)
        ignore_index: Index to ignore in loss computation (default: -100)
        teacher_data: List of teacher data dictionaries, one per batch element.
                     Each dict contains 'positions', 'indices', 'values' keys.
        temp: Temperature parameter for softmax (default: 1.0)
        chunk_size: Sequence chunk size for memory-efficient processing (default: 1024)
        debug: Enable debug output (default: False)
        
    Returns:
        - kl_losses: Per-token KL divergence losses of shape [B, T]
        
    Raises:
        ValueError: If teacher_data is None and cannot be generated
    """
    # Compute standard cross-entropy loss with log-sum-exp values
    cross_entropy_loss, log_sum_exp_values = cce_per_token_loss(
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
    # Transpose embeddings to batch-first format: [B, S, H]
    batch_first_embeddings = embeddings.transpose(0, 1).contiguous()
    
    # Get tensor parallelism configuration
    tensor_parallel_world_size = get_tensor_model_parallel_world_size()
    tensor_parallel_rank = get_tensor_model_parallel_rank()
    is_tensor_parallel = tensor_parallel_world_size > 1
    
    # Get vocabulary partition range for this rank
    vocab_start_idx, vocab_end_idx = VocabUtility.vocab_range_from_global_vocab_size(
        vocab_size, tensor_parallel_rank, tensor_parallel_world_size
    )
    
    device = batch_first_embeddings.device
    batch_size, sequence_length, hidden_size = batch_first_embeddings.shape

    # Initialize KL loss tensor
    kl_loss_tensor = torch.zeros(
        (batch_size, sequence_length), 
        device=device, 
        dtype=torch.float32
    )
    
    # Process each batch element separately
    for batch_idx in range(batch_size):
        _process_batch_element_kl_loss(
            batch_idx=batch_idx,
            batch_embeddings=batch_first_embeddings[batch_idx],
            batch_labels=labels[batch_idx],
            batch_log_sum_exp=log_sum_exp_values[batch_idx].view(-1),
            batch_teacher_data=teacher_data[batch_idx],
            classifier_weight=classifier_weight,
            kl_loss_tensor=kl_loss_tensor,
            vocab_start_idx=vocab_start_idx,
            vocab_end_idx=vocab_end_idx,
            is_tensor_parallel=is_tensor_parallel,
            temp=temp,
            chunk_size=chunk_size,
            debug=debug,
            ignore_index=ignore_index,
        )
    
    # Reduce KL losses across tensor parallel ranks
    if is_tensor_parallel:
        kl_loss_tensor = reduce_from_tensor_model_parallel_region(kl_loss_tensor)

    if debug:
        _print_debug_summary(kl_loss_tensor)

    return kl_loss_tensor


def _process_batch_element_kl_loss(
    batch_idx: int,
    batch_embeddings: torch.Tensor,
    batch_labels: torch.Tensor,
    batch_log_sum_exp: torch.Tensor,
    batch_teacher_data: Dict[str, Any],
    classifier_weight: torch.Tensor,
    kl_loss_tensor: torch.Tensor,
    vocab_start_idx: int,
    vocab_end_idx: int,
    is_tensor_parallel: bool,
    temp: float,
    chunk_size: int,
    debug: bool,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> None:
    """
    Process KL loss computation for a single batch element.
    
    Args:
        batch_idx: Index of current batch element
        batch_embeddings: Embeddings for this batch element [T, H]
        batch_labels: Labels for this batch element [T]
        batch_log_sum_exp: Log-sum-exp values for this batch element [T]
        batch_teacher_data: Teacher data for this batch element
        classifier_weight: Classifier weight matrix
        kl_loss_tensor: Output tensor for KL losses
        vocab_start_idx: Start index of vocabulary partition
        vocab_end_idx: End index of vocabulary partition
        is_tensor_parallel: Whether tensor parallelism is enabled
        temp: Temperature parameter
        chunk_size: Processing chunk size
        ignore_index: Index to ignore
        debug: Enable debug output
    """
    # Build teacher data lookup by position
    teacher_lookup_by_position = _build_teacher_lookup(
        batch_teacher_data, batch_embeddings.device
    )
    sequence_length = batch_embeddings.size(0)
    
    # Process sequence in chunks for memory efficiency
    for chunk_start in range(0, sequence_length, chunk_size):
        chunk_end = min(chunk_start + chunk_size, sequence_length)

        chunk_data = _extract_chunk_teacher_data(
            teacher_lookup_by_position=teacher_lookup_by_position,
            batch_labels=batch_labels,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            vocab_start_idx=vocab_start_idx,
            vocab_end_idx=vocab_end_idx,
            is_tensor_parallel=is_tensor_parallel,
            temp=temp,
            ignore_index=ignore_index
        )

        if not chunk_data['positions']:
            continue
            
        _compute_chunk_kl_losses(
            batch_idx=batch_idx,
            chunk_data=chunk_data,
            batch_embeddings=batch_embeddings,
            batch_log_sum_exp=batch_log_sum_exp,
            classifier_weight=classifier_weight,
            kl_loss_tensor=kl_loss_tensor,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            temp=temp,
            debug=debug,
        )
        

def _build_teacher_lookup(
    batch_teacher_data: Dict[str, Any], 
    device: torch.device
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Build a lookup dictionary mapping positions to teacher data.
    
    Args:
        batch_teacher_data: Teacher data for a batch element
        device: Target device for tensors
        
    Returns:
        Dictionary mapping position indices to (teacher_indices, teacher_values) tuples
    """
    teacher_lookup = {}
    
    positions = batch_teacher_data.get('positions', [])
    indices_list = batch_teacher_data.get('indices', [])
    values_list = batch_teacher_data.get('values', [])
    
    for position, teacher_indices, teacher_values in zip(positions, indices_list, values_list):
        position_idx = int(position.item() if isinstance(position, torch.Tensor) else position)
        
        teacher_indices_tensor = torch.as_tensor(
            teacher_indices, device=device, dtype=torch.long
        ).flatten()
        teacher_values_tensor = torch.as_tensor(
            teacher_values, device=device
        ).flatten()
        
        teacher_lookup[position_idx] = (teacher_indices_tensor, teacher_values_tensor)
    
    return teacher_lookup


def _extract_chunk_teacher_data(
    teacher_lookup_by_position: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    batch_labels: torch.Tensor,
    chunk_start: int,
    chunk_end: int,
    vocab_start_idx: int,
    vocab_end_idx: int,
    is_tensor_parallel: bool,
    temp: float,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> Dict[str, List]:
    """
    Extract and filter teacher data for a specific chunk.
    
    Args:
        teacher_lookup_by_position: Teacher data lookup by position
        batch_labels: Labels for the batch element
        chunk_start: Start index of chunk
        chunk_end: End index of chunk
        vocab_start_idx: Start of vocabulary partition
        vocab_end_idx: End of vocabulary partition
        is_tensor_parallel: Whether tensor parallelism is enabled
        temp: Temperature parameter
        ignore_index: Index to ignore
        
    Returns:
        Dictionary containing filtered chunk data
    """
    chunk_data = {
        'positions': [],
        'teacher_indices': [],
        'teacher_values': [],
        'positions_in_chunk': []
    }

    cp_world_size = get_context_parallel_world_size()
    offset = 0

    if cp_world_size > 1:
        local_sequence_length = batch_labels.size(0)
        cp_rank = get_context_parallel_rank()
        offset = cp_rank * local_sequence_length

    for position in range(chunk_start, chunk_end):

        if batch_labels[position] == ignore_index:
            continue

        global_position = offset + position

        teacher_entry = teacher_lookup_by_position.get(global_position)
        
        if teacher_entry is None:
            continue

        teacher_indices, teacher_values = teacher_entry

        # Apply temperature and compute log probabilities
        teacher_log_probs = F.log_softmax(
            teacher_values / temp, dim=0, dtype=torch.float32
        )
        
        # Filter for local vocabulary partition if using tensor parallelism
        if is_tensor_parallel:
            vocab_mask = (teacher_indices >= vocab_start_idx) & (teacher_indices < vocab_end_idx)
            if not vocab_mask.any():
                continue
            local_teacher_indices = teacher_indices[vocab_mask] - vocab_start_idx
            local_teacher_log_probs = teacher_log_probs[vocab_mask]
        else:
            local_teacher_indices = teacher_indices
            local_teacher_log_probs = teacher_log_probs

        chunk_data['positions'].append(position)
        chunk_data['teacher_indices'].append(local_teacher_indices)
        chunk_data['teacher_values'].append(local_teacher_log_probs)
        chunk_data['positions_in_chunk'].append(position - chunk_start)
    
    return chunk_data


def _compute_chunk_kl_losses(
    batch_idx: int,
    chunk_data: Dict[str, List],
    batch_embeddings: torch.Tensor,
    batch_log_sum_exp: torch.Tensor,
    classifier_weight: torch.Tensor,
    kl_loss_tensor: torch.Tensor,
    chunk_start: int,
    chunk_end: int,
    temp: float,
    debug: bool,
) -> None:
    """
    Compute KL divergence losses for a chunk of positions.
    
    Args:
        batch_idx: Batch element index
        chunk_data: Extracted chunk data
        batch_embeddings: Embeddings for the batch element
        batch_log_sum_exp: Log-sum-exp values for the batch element
        classifier_weight: Classifier weight matrix
        kl_loss_tensor: Output tensor for KL losses
        chunk_start: Start index of chunk
        chunk_end: End index of chunk
        temp: Temperature parameter
        debug: Enable debug output
    """
    device = batch_embeddings.device
    num_positions = len(chunk_data['positions'])
    
    # Convert positions in chunk to tensor
    positions_in_chunk_tensor = torch.tensor(
        chunk_data['positions_in_chunk'], device=device, dtype=torch.long
    )
    
    # Pad teacher data to uniform length for vectorized operations
    max_teacher_tokens = max(len(indices) for indices in chunk_data['teacher_indices'])
    
    # Create padded tensors for batch processing
    padded_teacher_indices = torch.full(
        (num_positions, max_teacher_tokens), -1, device=device, dtype=torch.long
    )
    padded_teacher_values = torch.zeros(
        (num_positions, max_teacher_tokens), device=device
    )
    teacher_mask = torch.zeros(
        (num_positions, max_teacher_tokens), device=device, dtype=torch.bool
    )
    
    # Fill padded tensors
    for i, (teacher_indices, teacher_log_probs) in enumerate(
        zip(chunk_data['teacher_indices'], chunk_data['teacher_values'])
    ):
        num_tokens = len(teacher_indices)
        padded_teacher_indices[i, :num_tokens] = teacher_indices
        padded_teacher_values[i, :num_tokens] = teacher_log_probs
        teacher_mask[i, :num_tokens] = True
    
    # Get unique teacher indices for efficient computation
    valid_teacher_indices = padded_teacher_indices[teacher_mask]
    if len(valid_teacher_indices) == 0:
        return
    
    unique_teacher_indices = torch.unique(valid_teacher_indices, sorted=True)
    
    # Compute student logits for union of teacher indices
    union_classifier_weights = classifier_weight.index_select(0, unique_teacher_indices)
    chunk_embeddings = batch_embeddings[chunk_start:chunk_end]
    chunk_logits = chunk_embeddings @ union_classifier_weights.t().contiguous()
    
    # Map teacher indices to positions in unique indices
    teacher_columns = torch.searchsorted(unique_teacher_indices, padded_teacher_indices)
    
    # Get student logits for positions with teacher data
    position_logits = chunk_logits[positions_in_chunk_tensor]
    gathered_student_logits = torch.gather(position_logits, 1, teacher_columns)
    
    # Apply temperature and compute student log probabilities
    scaled_student_logits = gathered_student_logits / temp
    position_log_sum_exp = batch_log_sum_exp[chunk_data['positions']].unsqueeze(1)
    student_log_probs = scaled_student_logits - position_log_sum_exp
    
    # Compute KL divergence: KL(teacher || student)
    # KL = sum(p_teacher * (log p_teacher - log p_student))
    kl_per_token = padded_teacher_values.exp() * (padded_teacher_values - student_log_probs)
    kl_per_token = kl_per_token * teacher_mask.float()  # Mask invalid positions
    kl_per_position = kl_per_token.sum(dim=1)
    
    # Apply temperature scaling
    kl_per_position_scaled = kl_per_position * (temp ** 2)
    
    # Store results in output tensor
    kl_loss_tensor[batch_idx, chunk_data['positions']] = kl_per_position_scaled
    
    if debug:
        _print_chunk_debug_info(
            batch_idx, chunk_data['positions'], padded_teacher_values, 
            student_log_probs, teacher_mask, position_log_sum_exp, kl_loss_tensor
        )


def _print_chunk_debug_info(
    batch_idx: int,
    positions: List[int],
    teacher_values: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_mask: torch.Tensor,
    log_sum_exp_values: torch.Tensor,
    kl_losses: torch.Tensor,
) -> None:
    """Print debug information for chunk processing."""
    debug_position = 3157
    if debug_position in positions:
        position_index = positions.index(debug_position)
        print(f"batch {batch_idx} efficient pos {debug_position} "
              f"teacher_values_batch: {teacher_values[position_index].exp()}")
        print(f"batch {batch_idx} efficient pos {debug_position} "
              f"s_logpK: {student_log_probs[position_index] * teacher_mask[position_index].float()} "
              f"with logZ {log_sum_exp_values[position_index]}")
        print(f"batch {batch_idx} efficient pos {debug_position} "
              f"kl_per_pos: {kl_losses[batch_idx, debug_position]}")


def _print_debug_summary(
    kl_losses: torch.Tensor, 
) -> None:
    """Print debug summary information."""
    print(f"  KL losses: {kl_losses.sum()}")


def generate_fake_teacher_data(
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    num_teacher_tokens_per_pos: int = 3,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> List[Dict[str, List]]:
    """
    Generate synthetic teacher data for testing distillation functionality.
    
    This function creates fake teacher data that mimics the structure expected
    by the distillation loss computation. Each batch element gets teacher data
    for all sequence positions.
    
    Args:
        batch_size: Number of batch elements
        seq_length: Length of each sequence
        vocab_size: Total vocabulary size
        num_teacher_tokens_per_pos: Number of teacher tokens per position (default: 3)
        device: Target device for tensors (default: CPU)
        dtype: Data type for teacher values (default: float32)
        
    Returns:
        List of teacher data dictionaries, one per batch element.
        Each dictionary contains:
        - 'positions': List of sequence positions
        - 'indices': List of teacher token indices for each position
        - 'values': List of teacher probability values for each position
    """
    if device is None:
        device = torch.device('cpu')
    
    teacher_data = []
    
    for batch_idx in range(batch_size):
        # Generate positions for all sequence positions
        sequence_positions = torch.arange(0, seq_length, device=device)
        
        batch_teacher_data = {
            'positions': [],
            'indices': [],
            'values': []
        }
        
        for position in sequence_positions:
            # Generate random vocabulary indices (sorted for consistency)
            random_teacher_indices, _ = torch.sort(
                torch.randint(0, vocab_size, (num_teacher_tokens_per_pos,), device=device)
            )
            
            # Generate random values that will be normalized to probabilities
            random_teacher_values = torch.rand(
                num_teacher_tokens_per_pos, device=device, dtype=dtype
            )
            
            batch_teacher_data['positions'].append(position)
            batch_teacher_data['indices'].append(random_teacher_indices)
            batch_teacher_data['values'].append(random_teacher_values)
        
        teacher_data.append(batch_teacher_data)
    
    return teacher_data


def traditional_distillation_loss(
    student_logits: torch.Tensor,
    teacher_data: List[Dict[str, Any]],
    labels: torch.Tensor,
    T: float = DEFAULT_TEMPERATURE,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> torch.Tensor:
    """
    Compute knowledge distillation loss using traditional approach.
    
    This function implements the standard Knowledge Distillation approach
    from Hinton et al. (2015), computing KL divergence between teacher
    and student distributions at specified temperature.
    
    The loss is computed as KL(P_teacher || Q_student) where:
    - P_teacher: Softmax over teacher-provided logits at temperature T
    - Q_student: Student's log-softmax over full vocabulary at temperature T,
                 gathered at teacher indices (normalization over full vocab)
    
    The KD term is scaled by T^2 as recommended in the literature.
    
    Args:
        student_logits: Student model logits of shape (B, T, V)
        teacher_data: List of teacher data dictionaries, one per batch element.
                     Each dict contains 'positions', 'indices', 'values' keys:
                     - positions: List of sequence positions with teacher data
                     - indices: List of teacher token indices for each position  
                     - values: List of teacher logit values for each position
        labels: Target labels of shape (B, T) with ignore_index for masked positions
        T: Temperature parameter for softmax (default: 1.0)
        ignore_index: Index value to ignore in loss computation (default: -100)
        
    Returns:
        Scalar KL divergence loss value, averaged over valid positions
        and scaled by temperature squared
        
    Raises:
        ValueError: If teacher_data length doesn't match batch size
    """
    # Transpose to batch-first format: (B, T, V)
    student_logits_batch_first = student_logits.transpose(0, 1).contiguous()
    device = student_logits_batch_first.device
    batch_size, sequence_length, vocab_size = student_logits_batch_first.shape
    if len(teacher_data) != batch_size:
        raise ValueError(
            f"teacher_data length {len(teacher_data)} must match batch size {batch_size}"
        )
    
    # Initialize accumulators
    total_kl_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_valid_positions = torch.zeros((), device=device, dtype=torch.float32)
    
    # Debug configuration
    debug_position_idx = 3157
    
    cp_world_size = get_context_parallel_world_size()
    cp_rank = get_context_parallel_rank() if cp_world_size > 1 else 0
    offset = cp_rank * sequence_length

    def _map_global_to_local_position(global_position: int) -> Optional[int]:
        """Map a teacher position to the local index for this CP rank."""
        return global_position - offset

    # Process each batch element independently
    for batch_idx in range(batch_size):
        batch_teacher_data = teacher_data[batch_idx]
        batch_label_sequence = labels[batch_idx]  # Shape: (T,)
        batch_student_logits = student_logits_batch_first[batch_idx]  # Shape: (T, V)
        
        # Extract teacher data components
        teacher_positions = batch_teacher_data.get('positions', [])
        teacher_indices_list = batch_teacher_data.get('indices', [])
        teacher_values_list = batch_teacher_data.get('values', [])
        
        if not teacher_positions or not teacher_indices_list or not teacher_values_list:
            continue  # Skip batch elements without teacher data
        
        batch_kl_accumulator = torch.zeros_like(total_kl_loss)
        batch_valid_positions = torch.zeros_like(total_valid_positions)
        
        # Process each position with teacher data
        for teacher_position, teacher_token_indices, teacher_token_values in zip(
            teacher_positions, teacher_indices_list, teacher_values_list
        ):
            position_idx = int(
                teacher_position.item() if isinstance(teacher_position, torch.Tensor) 
                else teacher_position
            )

            local_position_idx = _map_global_to_local_position(position_idx)
            if local_position_idx >= sequence_length or batch_label_sequence[local_position_idx] == ignore_index:
                continue

            # Convert teacher data to tensors on correct device
            teacher_indices_tensor = torch.as_tensor(
                teacher_token_indices, dtype=torch.long, device=device
            )
            teacher_values_tensor = torch.as_tensor(teacher_token_values, device=device)
            
            # Get student logits for this position
            student_logits_at_position = batch_student_logits[local_position_idx]  # Shape: (V,)
            
            # Compute student log-probabilities with temperature scaling
            # Normalization is over the full vocabulary
            student_log_probs_full_vocab = F.log_softmax(
                    student_logits_at_position / T, dim=-1, dtype=torch.float32
                ) # Shape: (V,)
              
            
            # Extract student log-probabilities for teacher token indices
            student_log_probs_teacher_tokens = student_log_probs_full_vocab[teacher_indices_tensor]  # Shape: (K,)
            
            # Compute teacher log-probabilities with temperature scaling
            # Normalization is over teacher indices only
            teacher_log_probs = F.log_softmax(
                teacher_values_tensor / T, dim=-1, dtype=torch.float32
            )  # Shape: (K,)
            
            # Compute KL divergence: KL(P_teacher || Q_student)
            # KL = sum(p_teacher * (log p_teacher - log p_student))
            teacher_probs = teacher_log_probs.exp()
            kl_divergence_at_position = teacher_probs * (teacher_log_probs - student_log_probs_teacher_tokens)
            kl_divergence_scalar = kl_divergence_at_position.sum() * (T ** 2)  # Reduce to scalar

            # Debug output for specific position
            if position_idx == debug_position_idx:
                full_vocab_log_sum_exp = torch.logsumexp(
                    student_logits_at_position.float() / T, dim=-1
                )
                print(f"batch {batch_idx} traditional pos {debug_position_idx} (local {local_position_idx}) "
                      f"teacher_probs: {teacher_probs}")
                print(f"batch {batch_idx} traditional pos {debug_position_idx} (local {local_position_idx}) "
                      f"student_log_probs_teacher: {student_log_probs_teacher_tokens} "
                      f"lse student: {full_vocab_log_sum_exp}")
                print(f"batch {batch_idx} traditional pos {debug_position_idx} (local {local_position_idx}) "
                      f"kl loss {kl_divergence_scalar}")
            
            batch_kl_accumulator += kl_divergence_scalar 
            batch_valid_positions += 1
        
        total_kl_loss += batch_kl_accumulator
        total_valid_positions += batch_valid_positions
        
        print(f"batch {batch_idx} traditional valid positions: {batch_valid_positions}, "
              f"batch_kl_sum: {batch_kl_accumulator}")
    
    # Compute average loss over valid positions with temperature scaling
    if total_valid_positions > 0:
        knowledge_distillation_loss = (total_kl_loss) 
    else:
        knowledge_distillation_loss = torch.zeros_like(total_kl_loss)
    
    print(f"KD loss: {knowledge_distillation_loss}, valid positions: {total_valid_positions}")
    return knowledge_distillation_loss
