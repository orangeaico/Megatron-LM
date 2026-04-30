# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from typing import Any, Dict, Optional

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split

IGNORE_INDEX = -100


class SFTLowLevelDataset:
    """The low-level dataset loading jsonl data for SFT

    Args:
        dataset_path (str): The path to jsonl data
            Each line of the jsonl must have key "messages" (List[Dict]),
            which is a sequence of system/user/assistant messages.
            Must be in the following format:
            [
                {"role": "system", "content": "something"},
                {"role": "user", "content": "something1"},
                {"role": "assistant", "content": "something2"},
            ]
            A jsonl line can contain multiple conversations packed together into on list. Each
            conversation starts with the system role, and conversations can have multiple turns
            of the user and assistant roles.
    """

    def __init__(self, dataset_path: str) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "SFTDataset currently requires datasets library to be installed"
            )
        self.dataset = load_dataset("json", data_files=dataset_path, split="all")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> list:
        if "conversations" in self.dataset[idx]:
            return self.dataset[idx]["conversations"]
        if "messages" in self.dataset[idx]:
            return self.dataset[idx]["messages"]
        raise ValueError(
            f"The sample must have 'conversations' or 'messages' but got {self.dataset[idx]}"
        )


class SFTDataset(MegatronDataset):
    """The dataset used during SFT"""

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        return SFTLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        return self.num_samples

    @staticmethod
    def _as_token_ids(tokenized):
        if hasattr(tokenized, "input_ids"):
            return SFTDataset._as_token_ids(tokenized.input_ids)
        if hasattr(tokenized, "ids"):
            return list(tokenized.ids)
        if isinstance(tokenized, np.ndarray):
            if tokenized.ndim == 2:
                if tokenized.shape[0] != 1:
                    raise ValueError(f"Expected one tokenized sequence, got {tokenized.shape}")
                tokenized = tokenized[0]
            return tokenized.astype(np.int64, copy=False).tolist()
        if isinstance(tokenized, (list, tuple)):
            if len(tokenized) == 1 and not isinstance(tokenized[0], (int, np.integer)):
                return SFTDataset._as_token_ids(tokenized[0])
            return list(tokenized)
        raise TypeError(f"Unsupported tokenized message type: {type(tokenized)}")

    def _process_example(self, tokenizer, conversation: list) -> tuple[list[int], list[int]]:
        if not isinstance(conversation, list):
            raise ValueError(f"The sample must be a list but got {type(conversation)}")

        input_ids = []
        labels = []
        for message in conversation:
            seg_ids = self._as_token_ids(
                tokenizer.apply_chat_template(
                    [message],
                    tokenize=True,
                    add_generation_prompt=False,
                )
            )
            input_ids.extend(seg_ids)
            if message["role"].lower() == "assistant":
                labels.extend(seg_ids)
            else:
                labels.extend([IGNORE_INDEX] * len(seg_ids))

        assert len(input_ids) == len(labels)
        return input_ids, labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        tokenizer = self.config.tokenizer
        max_seq_len = self.config.sequence_length
        pad = tokenizer.pad

        conversation = self.dataset[int(self.indices[idx % len(self.indices)])]
        tokens, labels = self._process_example(tokenizer, conversation)
        labels = labels[1:] + [IGNORE_INDEX]

        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
            labels = labels[:max_seq_len]

        padding_len = max_seq_len - len(tokens)
        tokens = tokens + [pad] * padding_len
        labels = labels + [IGNORE_INDEX] * padding_len

        input_ids = torch.tensor(tokens, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)
        position_ids = torch.arange(max_seq_len, dtype=torch.int64)

        loss_mask = torch.ones(max_seq_len, dtype=torch.float32)
        loss_mask[labels == pad] = 0.0
        loss_mask[labels == IGNORE_INDEX] = 0.0

        ret = {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
        }

        if self.config.create_attention_mask:
            attention_mask = torch.tril(torch.ones((max_seq_len, max_seq_len))).unsqueeze(0)
            ret['attention_mask'] = attention_mask < 0.5

        return ret
