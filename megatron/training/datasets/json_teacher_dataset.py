"""Dataset utilities for JSON files containing sparse teacher logits payloads."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split


_LABEL_PAD_ID = -100
DEBUG = False


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
    
    if dropped and DEBUG:
        print(f"Dropped {dropped} teacher logits positions outside valid range [0, {seq_length})")

    if not normalized_positions:
        return None

    return {
        "positions": normalized_positions,
        "indices": normalized_indices,
        "values": normalized_values,
    }


def _convert_teacher_payload_to_tensors(
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Convert normalized teacher payload lists into torch tensors on CPU."""
    positions = torch.as_tensor(payload["positions"], dtype=torch.long).contiguous()

    indices_tensors = [
        torch.as_tensor(entry, dtype=torch.long).view(-1).contiguous()
        for entry in payload["indices"]
    ]
    values_tensors = [
        torch.as_tensor(entry, dtype=torch.float32).view(-1).contiguous()
        for entry in payload["values"]
    ]

    if not indices_tensors:
        row_ptr = torch.zeros(1, dtype=torch.long)
        flat_indices = torch.empty(0, dtype=torch.long)
        flat_values = torch.empty(0, dtype=torch.float32)
    else:
        lengths = torch.as_tensor(
            [tensor.numel() for tensor in indices_tensors], dtype=torch.long
        )
        row_ptr = torch.zeros(len(indices_tensors) + 1, dtype=torch.long)
        row_ptr[1:] = torch.cumsum(lengths, dim=0)
        flat_indices = torch.cat(indices_tensors, dim=0)
        flat_values = torch.cat(values_tensors, dim=0)

    return {
        "positions": positions,
        "row_ptr": row_ptr,
        "indices": flat_indices,
        "values": flat_values,
    }

class JsonTeacherLowLevelDataset:
    """Minimal iterable over JSON teacher records."""

    def __init__(self, dataset_path: str) -> None:
        if not os.path.isdir(dataset_path):
            raise ValueError(f"JSON teacher data directory '{dataset_path}' does not exist")

        file_paths = [
            os.path.join(dataset_path, entry)
            for entry in os.listdir(dataset_path)
            if entry.endswith(".json")
        ]

        if not file_paths:
            raise ValueError(f"No JSON files found in directory '{dataset_path}'")

        self.file_paths = file_paths

    def __len__(self) -> int:
        print(f"Number of JSON files found: {len(self.file_paths)}")
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_path = self.file_paths[idx]
        with open(file_path, "r", encoding="utf-8") as reader:
            record = json.load(reader)
        return record


class JsonTeacherDataset(MegatronDataset):
    """Dataset that reads one JSON file per training example."""

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
        *,
        label_pad_id: int = _LABEL_PAD_ID,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

        if self.config.tokenizer is None:
            raise ValueError("JsonTeacherDataset requires a tokenizer in the config")

        self.seq_length = self.config.sequence_length
        self.label_pad_id = label_pad_id
        self.create_attention_mask = self.config.create_attention_mask

        tokenizer = self.config.tokenizer

        pad_token = getattr(tokenizer, "pad", None)
        eod_token = getattr(tokenizer, "eod", None)

        if pad_token is None and eod_token is None:
            raise ValueError("Tokenizer must define either 'pad' or 'eod' token id")

        if pad_token is None:
            pad_token = eod_token
        if eod_token is None:
            eod_token = pad_token

        self.pad_token_id = int(pad_token)
        self.eod_token_id = int(eod_token)

        if self.create_attention_mask:
            mask = torch.tril(torch.ones(self.seq_length, self.seq_length, dtype=torch.bool))
            self._attention_mask_template = mask.unsqueeze(0)  # [1, S, S]
        else:
            self._attention_mask_template = None

        self._position_ids = torch.arange(self.seq_length, dtype=torch.long)
        self.collate_fn = JsonTeacherCollator(create_attention_mask=self.create_attention_mask)

        if len(self.indices) == 0:
            self.num_samples = 0

    def __len__(self) -> int:
        if self.num_samples is not None:
            return self.num_samples
        return len(self.indices)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        return JsonTeacherLowLevelDataset(dataset_path)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dataset_length = len(self.indices)
        if dataset_length == 0:
            raise IndexError("JsonTeacherDataset has no indices to sample")

        actual_index = int(self.indices[idx % dataset_length])
        record = self.dataset[actual_index]

        file_path = None
        if hasattr(self.dataset, "file_paths"):
            file_path = self.dataset.file_paths[actual_index]  # type: ignore[attr-defined]

        if "input_ids" not in record or "labels" not in record:
            location = f" '{file_path}'" if file_path else ""
            raise KeyError(
                f"Record{location} must contain 'input_ids' and 'labels' fields"
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
        labels = torch.cat([labels[1:], torch.tensor([self.eod_token_id])])

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

        position_ids = self._position_ids.clone()

        teacher_raw = record.get("teacher_logits") or record.get("teacher_data")
        teacher_data = _normalize_teacher_payload(teacher_raw, self.seq_length)
        if teacher_data is not None:
            teacher_data = _convert_teacher_payload_to_tensors(teacher_data)

        if DEBUG:
            print(f"Loaded record from {file_path if file_path else 'in-memory'}")

        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "teacher_data": teacher_data,
        }


class JsonTeacherCollator:
    """Collate function that stacks tensors and keeps teacher payload per sample."""

    def __init__(self, create_attention_mask: bool) -> None:
        self.create_attention_mask = create_attention_mask

    def __call__(self, samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        sample_list = list(samples)
        if not sample_list:
            raise ValueError("JsonTeacherCollator received an empty batch")

        tokens = torch.stack([sample["tokens"] for sample in sample_list])
        labels = torch.stack([sample["labels"] for sample in sample_list])
        loss_mask = torch.stack([sample["loss_mask"] for sample in sample_list])
        position_ids = torch.stack([sample["position_ids"] for sample in sample_list])

        if self.create_attention_mask:
            attention = torch.stack([
                sample["attention_mask"] for sample in sample_list
            ])
        else:
            attention = None

        teacher_list = [sample["teacher_data"] for sample in sample_list]

        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "attention_mask": attention,
            "teacher_data": teacher_list,
        }
