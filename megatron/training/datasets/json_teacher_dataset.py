"""Dataset utilities for JSON files containing sparse teacher logits payloads."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from megatron.core import mpu


_LABEL_PAD_ID = -100


def _to_sequence(value: Any) -> List[Any]:
    """Normalize a JSON field into a Python list."""
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().view(-1).tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _pad_or_trim_tensor(
    data: Sequence[Any],
    length: int,
    pad_value: Any,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.as_tensor(data, dtype=dtype)
    if tensor.numel() < length:
        pad_size = length - tensor.numel()
        pad_tensor = torch.full((pad_size,), pad_value, dtype=dtype)
        tensor = torch.cat((tensor, pad_tensor), dim=0)
    elif tensor.numel() > length:
        tensor = tensor[:length]
    return tensor.contiguous()


def _normalize_teacher_payload(
    payload: Optional[Dict[str, Any]],
    seq_length: int,
) -> Optional[Dict[str, List[List[float]]]]:
    if payload is None:
        return None

    positions = _to_sequence(payload.get("positions"))
    indices = payload.get("indices", [])
    values = payload.get("values", [])

    if not positions:
        return None

    if len(indices) != len(values) or len(indices) != len(positions):
        raise ValueError(
            "teacher_logits must provide matching lengths for 'positions', 'indices', and 'values'"
        )

    normalized_positions: List[int] = []
    normalized_indices: List[List[int]] = []
    normalized_values: List[List[float]] = []
    dropped = 0

    for position, idx_list, val_list in zip(positions, indices, values):
        pos_int = int(position)
        if pos_int < 0 or pos_int >= seq_length:
            dropped += 1
            continue
        idxs = [int(idx) for idx in _to_sequence(idx_list)]
        vals = [float(val) for val in _to_sequence(val_list)]
        if len(idxs) != len(vals):
            raise ValueError("teacher logits indices/values length mismatch per position")
        normalized_positions.append(pos_int)
        normalized_indices.append(idxs)
        normalized_values.append(vals)

    if not normalized_positions:
        if dropped:
            print(f"Dropped {dropped} teacher logits positions outside valid range [0, {seq_length})")
        return None

    if dropped:
        print(f"Dropped {dropped} teacher logits positions outside valid range [0, {seq_length})")

    return {
        "positions": normalized_positions,
        "indices": normalized_indices,
        "values": normalized_values,
    }


@dataclass
class JsonTeacherSample:
    tokens: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    teacher_data: Optional[Dict[str, List[List[float]]]]


class JsonTeacherDataset(Dataset):
    """Dataset that reads one JSON file per training example."""

    def __init__(
        self,
        data_dir: str,
        seq_length: int,
        *,
        pad_token_id: int,
        label_pad_id: int = _LABEL_PAD_ID,
        create_attention_mask: bool = True,
    ) -> None:
        self.seq_length = seq_length
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id
        self.create_attention_mask = create_attention_mask

        if not os.path.isdir(data_dir):
            raise ValueError(f"JSON teacher data directory '{data_dir}' does not exist")

        self.file_paths = sorted(
            os.path.join(data_dir, entry)
            for entry in os.listdir(data_dir)
            if entry.endswith(".json")
        )
        if not self.file_paths:
            raise ValueError(f"No JSON files found in directory '{data_dir}'")

        base_mask = None
        if create_attention_mask:
            mask = torch.tril(torch.ones(seq_length, seq_length, dtype=torch.bool))
            base_mask = mask.unsqueeze(0)  # [1, S, S]
        self._attention_mask_template = base_mask

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> JsonTeacherSample:
        file_path = self.file_paths[index]
        with open(file_path, "r", encoding="utf-8") as reader:
            record = json.load(reader)

        if "input_ids" not in record or "labels" not in record:
            raise KeyError(
                f"Record '{file_path}' must contain 'input_ids' and 'labels' fields"
            )

        tokens = _pad_or_trim_tensor(
            record["input_ids"],
            self.seq_length,
            self.pad_token_id,
            dtype=torch.long,
        )
        labels = _pad_or_trim_tensor(
            record["labels"],
            self.seq_length,
            self.label_pad_id,
            dtype=torch.long,
        )

        if "loss_mask" in record:
            loss_mask = _pad_or_trim_tensor(
                record["loss_mask"],
                self.seq_length,
                0.0,
                dtype=torch.float32,
            )
        else:
            loss_mask = (labels != self.label_pad_id).to(torch.float32)

        if self._attention_mask_template is not None:
            attention_mask = self._attention_mask_template.clone()
        else:
            attention_mask = None

        position_ids = torch.arange(self.seq_length, dtype=torch.long)

        teacher_raw = record.get("teacher_logits") or record.get("teacher_data")
        teacher_data = _normalize_teacher_payload(teacher_raw, self.seq_length)

        return JsonTeacherSample(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            position_ids=position_ids,
            attention_mask=attention_mask,
            teacher_data=teacher_data,
        )


class JsonTeacherCollator:
    """Collate function that stacks tensors and keeps teacher payload per sample."""

    def __init__(self, create_attention_mask: bool) -> None:
        self.create_attention_mask = create_attention_mask

    def __call__(self, samples: Iterable[JsonTeacherSample]) -> Dict[str, Any]:
        sample_list = list(samples)
        if not sample_list:
            raise ValueError("JsonTeacherCollator received an empty batch")

        tokens = torch.stack([sample.tokens for sample in sample_list])
        labels = torch.stack([sample.labels for sample in sample_list])
        loss_mask = torch.stack([sample.loss_mask for sample in sample_list])
        position_ids = torch.stack([sample.position_ids for sample in sample_list])

        if self.create_attention_mask:
            attention = torch.stack([
                sample.attention_mask for sample in sample_list
            ])  # type: ignore[arg-type]
        else:
            attention = None

        teacher_list = [sample.teacher_data for sample in sample_list]

        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "attention_mask": attention,
            "teacher_data": teacher_list,
        }


def build_json_teacher_dataloader(
    data_dir: str,
    *,
    seq_length: int,
    micro_batch_size: int,
    pad_token_id: int,
    label_pad_id: int = _LABEL_PAD_ID,
    create_attention_mask: bool,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    dataset = JsonTeacherDataset(
        data_dir,
        seq_length,
        pad_token_id=pad_token_id,
        label_pad_id=label_pad_id,
        create_attention_mask=create_attention_mask,
    )

    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=mpu.get_data_parallel_world_size(),
            rank=mpu.get_data_parallel_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
        )
        shuffle_flag = False
    else:
        sampler = None
        shuffle_flag = shuffle

    collate = JsonTeacherCollator(create_attention_mask=create_attention_mask)

    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate,
    )
