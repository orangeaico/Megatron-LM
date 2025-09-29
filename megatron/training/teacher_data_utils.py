"""Utilities for broadcasting teacher logits alongside Megatron batches."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


_POS_PAD = -1
_IDX_PAD = -1


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().view(-1).tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def pack_teacher_batch(
    teacher_batch: Optional[Sequence[Optional[Dict[str, Any]]]],
    seq_length: int,
    device: torch.device,
) -> Tuple[Optional[Dict[str, torch.Tensor]], Tuple[int, int]]:
    """Pad a batch of sparse teacher logits for tensor broadcast.

    Returns a dictionary of tensors and a tuple ``(max_positions, max_k)`` describing
    the padded dimensions. ``None`` signals that no teacher data is present.
    """
    if not teacher_batch:
        return None, (-1, -1)

    batch_list: List[Dict[str, Any]] = []
    max_positions = 0
    max_k = 0
    for entry in teacher_batch:
        entry_dict = {
            "positions": [],
            "indices": [],
            "values": [],
        }
        if entry:
            positions_list = _to_list(entry.get("positions"))
            indices_list = entry.get("indices", [])
            values_list = entry.get("values", [])
            if len(indices_list) != len(values_list):
                raise ValueError("teacher_data 'indices' and 'values' length mismatch")
            if len(indices_list) != len(positions_list):
                raise ValueError("teacher_data 'positions' must align with 'indices'/'values'")
            for pos, idxs, vals in zip(positions_list, indices_list, values_list):
                pos_int = int(pos)
                if pos_int < 0 or pos_int >= seq_length:
                    raise ValueError(f"teacher position {pos_int} out of range [0, {seq_length})")
                idxs_list = _to_list(idxs)
                vals_list = _to_list(vals)
                if len(idxs_list) != len(vals_list):
                    raise ValueError("teacher_data indices/values length mismatch per position")
                entry_dict["positions"].append(pos_int)
                entry_dict["indices"].append([int(i) for i in idxs_list])
                entry_dict["values"].append([float(v) for v in vals_list])
                max_k = max(max_k, len(idxs_list))
        max_positions = max(max_positions, len(entry_dict["positions"]))
        batch_list.append(entry_dict)

    if max_positions == 0 or max_k == 0:
        return None, (-1, -1)

    batch_size = len(batch_list)
    positions = torch.full(
        (batch_size, max_positions),
        _POS_PAD,
        dtype=torch.int64,
        device=device,
    )
    indices = torch.full(
        (batch_size, max_positions, max_k),
        _IDX_PAD,
        dtype=torch.int64,
        device=device,
    )
    values = torch.zeros(
        (batch_size, max_positions, max_k),
        dtype=torch.float32,
        device=device,
    )
    counts = torch.zeros((batch_size, max_positions), dtype=torch.int64, device=device)

    for b_idx, entry in enumerate(batch_list):
        for pos_idx, (pos, idxs, vals) in enumerate(
            zip(entry["positions"], entry["indices"], entry["values"])
        ):
            k = len(idxs)
            positions[b_idx, pos_idx] = pos
            counts[b_idx, pos_idx] = k
            if k:
                indices[b_idx, pos_idx, :k] = torch.as_tensor(idxs, dtype=torch.int64, device=device)
                values[b_idx, pos_idx, :k] = torch.as_tensor(vals, dtype=torch.float32, device=device)

    return {
        "positions": positions,
        "indices": indices,
        "values": values,
        "counts": counts,
    }, (max_positions, max_k)


def allocate_teacher_tensors(
    batch_size: int,
    max_positions: int,
    max_k: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    positions = torch.full(
        (batch_size, max_positions),
        _POS_PAD,
        dtype=torch.int64,
        device=device,
    )
    indices = torch.full(
        (batch_size, max_positions, max_k),
        _IDX_PAD,
        dtype=torch.int64,
        device=device,
    )
    values = torch.zeros(
        (batch_size, max_positions, max_k),
        dtype=torch.float32,
        device=device,
    )
    counts = torch.zeros((batch_size, max_positions), dtype=torch.int64, device=device)
    return {
        "positions": positions,
        "indices": indices,
        "values": values,
        "counts": counts,
    }


def unpack_teacher_batch(packed: Optional[Dict[str, torch.Tensor]]) -> Optional[List[Dict[str, torch.Tensor]]]:
    if packed is None:
        return None

    positions = packed["positions"].detach().cpu()
    indices = packed["indices"].detach().cpu()
    values = packed["values"].detach().cpu()
    counts = packed["counts"].detach().cpu()

    batch: List[Dict[str, torch.Tensor]] = []
    batch_size = positions.size(0)
    max_positions = positions.size(1)
    for b_idx in range(batch_size):
        sample = {"positions": [], "indices": [], "values": []}
        for pos_idx in range(max_positions):
            count = counts[b_idx, pos_idx].item()
            if count <= 0:
                continue
            pos = positions[b_idx, pos_idx].item()
            sample["positions"].append(torch.tensor(pos, dtype=torch.int64))
            sample["indices"].append(
                indices[b_idx, pos_idx, :count].to(torch.int64)
            )
            sample["values"].append(
                values[b_idx, pos_idx, :count].to(torch.float32)
            )
        batch.append(sample)
    return batch


def has_teacher_data(packed: Optional[Dict[str, torch.Tensor]]) -> bool:
    if packed is None:
        return False
    counts = packed["counts"]
    return bool(counts.numel() and torch.any(counts > 0))
