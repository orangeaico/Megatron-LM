# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from typing import Any, Dict, Optional
from math import gcd
import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split

from megatron.core import parallel_state

IGNORE_INDEX = -100
DEBUG = True


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
        elif "messages" in self.dataset[idx]:
            return self.dataset[idx]["messages"]
        else:
            raise ValueError(f"The sample must have 'conversations' or 'messages' but got {self.dataset[idx]}")


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
    def _get_eos_id(tokenizer):
        hf_tokenizer = tokenizer._tokenizer
        if hf_tokenizer.eos_token == "<|eot_id|>":
            return 128001
        if hf_tokenizer.eos_token == "<|eot|>":
            return 200001
        if hf_tokenizer.eos_token == "<|im_end|>":
            return 151643

        return hf_tokenizer.eos_token_id

    def _process_example(self, tokenizer, conversation_list: Dict[str, Any]):        
        if not isinstance(conversation_list, list):
            raise ValueError(f"The sample must be a list but got {type(conversation_list)}")

        input_ids = []
        labels = []
        hf_tokenizer = tokenizer._tokenizer

        # Tokenize message-by-message using the chat template so formatting stays consistent
        for i, m in enumerate(conversation_list):
            seg_ids = hf_tokenizer.apply_chat_template(
                [m], tokenize=True, add_generation_prompt=False
            )
            input_ids.extend(seg_ids)
            if m["role"] == "assistant":
                labels.extend(seg_ids)                
            else:
                labels.extend([IGNORE_INDEX] * len(seg_ids))

        # Always add EOS between samples, but don’t train on it
        # input_ids = input_ids + [self._get_eos_id(tokenizer)]
        # labels = labels + [self._get_eos_id(tokenizer)]

        assert len(input_ids) == len(labels)
    
        return input_ids, labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        tokenizer = self.config.tokenizer
        max_seq_len = self.config.sequence_length

        conversation_list = self.dataset[int(self.indices[idx % len(self.indices)])]
        
        # print ("conversation_list: ", conversation_list[0])
        # tokens, target = tokenizer.tokenize_conversation(
        #     conversation_list, return_target=True, add_generation_prompt=False
        # )
        tokens, target = self._process_example(tokenizer, conversation_list)
        target = target[1:] + [IGNORE_INDEX]

        original_seq_len = len(tokens)

        if original_seq_len > max_seq_len:
            if True:  # TODO: when too long to fit in context, truncate left to right
                tokens = tokens[: max_seq_len]
                target = target[: max_seq_len]
            else:  # right to left
                tokens = tokens[-(max_seq_len - 1) :]
                target = target[-(max_seq_len - 1) :]

        use_variable_seq_len = getattr(self.config, "variable_seq_lengths", False)

        if use_variable_seq_len:
            def _lcm(a: int, b: int) -> int:
                if a == 0 or b == 0:
                    return max(a, b)
                return abs(a * b) // gcd(a, b)

            base_seq_len = len(tokens) 
            required_multiple = 1

            cp_size = parallel_state.get_context_parallel_world_size()
            if cp_size > 1:
                required_multiple = _lcm(required_multiple, 2 * cp_size)

            tp_size = 1
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                try:
                    tp_size = parallel_state.get_tensor_model_parallel_world_size()
                except AssertionError:
                    tp_size = 1
            if tp_size > 1:
                required_multiple = _lcm(required_multiple, tp_size)

            padding_len = 0
            if required_multiple > 1:
                remainder = base_seq_len % required_multiple
                if remainder != 0:
                    padding_len = required_multiple - remainder

            final_seq_len = base_seq_len + padding_len
            if final_seq_len > max_seq_len:
                # Trim tokens until both constraints fit within max_seq_len.
                while final_seq_len > max_seq_len and tokens:
                    tokens.pop()
                    target.pop()
                    base_seq_len = len(tokens)
                    padding_len = 0
                    if required_multiple > 1:
                        remainder = base_seq_len % required_multiple
                        if remainder != 0:
                            padding_len = required_multiple - remainder
                    final_seq_len = base_seq_len + padding_len
                if final_seq_len > max_seq_len:
                    raise ValueError(
                        "Unable to satisfy tensor/context parallel padding within max_seq_len"
                    )
        else:
            num_tokens = len(tokens)
            padding_len = max_seq_len - num_tokens
            if padding_len < 0:
                raise ValueError("Sample longer than configured sequence length after truncation")

        tokens = np.array(
            tokens + [tokenizer.pad] * padding_len,
            dtype=np.int64,
        )
        target = np.array(
            target + [IGNORE_INDEX] * padding_len,
            dtype=np.int64,
        )

        tokens = torch.tensor(tokens).contiguous()
        target = torch.tensor(target).contiguous()

        loss_mask, position_ids, attention_mask = self._get_ltor_masks_and_position_ids(
            max_seq_len, target, tokenizer.pad, use_variable_seq_len
        )

        if DEBUG:
            curr_rank = torch.distributed.get_rank()
            S = target.shape[0]
            trainable = int(loss_mask.sum().item())

            # Prefer deriving padding from the attention mask (robust when pad token == eos, or packed data)
            pad_tokens = 0
            nonpad_lengths = [S]  # default: assume fully non-padded

            if attention_mask is not None and attention_mask.dim() == 4:
                # attention_mask: [B, 1, S, S]; a "valid" token row typically has any True in its row
                vis = attention_mask.squeeze(1).any(dim=-1)  # [B, S] boolean
                nonpad_lengths = vis.sum(dim=0).tolist()     # per-sample non-padded token counts
                pad_tokens = int((S - vis.sum(dim=0)).sum().item())
            else:
                # Fallback: count pad tokens by id (only if pad_token_id is defined)
                pad_id = tokenizer.pad
                pad_mask = (tokens == pad_id)
                nonpad_lengths = (S - pad_mask.sum(dim=0)).tolist()
                pad_tokens = int(pad_mask.sum().item())

            pad_ratio = pad_tokens / S

            print(
                f"[Rank {curr_rank}][DATA_DEBUG] "
                f"Index {idx} | Sample Len={S} | Nonpad Lengths={nonpad_lengths} | "
                f"Original Seq len={original_seq_len} | Truncated S={S} | trainable_tokens={trainable} ({trainable/S:.2%}) "
                f"| nonpad_len={nonpad_lengths} | pad_ratio={pad_ratio:.2%}"
            )

        if self.config.create_attention_mask:
            ret = {
                'tokens': tokens,
                'labels': target,
                'attention_mask': attention_mask,
                'loss_mask': loss_mask,
                'position_ids': position_ids,
            }
        else:
            ret = {
                'tokens': tokens,
                'labels': target,
                'loss_mask': loss_mask,
                'position_ids': position_ids,
            }

        return ret

    def _get_ltor_masks_and_position_ids(self, max_seq_len, target, pad_token, use_variable_seq_len):
        """Build masks and position id for left to right model for SFT"""

        assert not self.config.reset_position_ids and not self.config.reset_attention_mask

        if use_variable_seq_len:
            seq_length = target.size(0)
        else:
            seq_length = max_seq_len
        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long)

        # Loss mask.
        loss_mask = torch.ones(seq_length, dtype=torch.float)
        loss_mask[target == pad_token] = 0.0  # mask paddings
        loss_mask[target == IGNORE_INDEX] = 0.0  # mask prompts

        if self.config.create_attention_mask:
            attention_mask = torch.tril(
                torch.ones((seq_length, seq_length), device=target.device)
            ).unsqueeze(0)
            # Convert attention mask to binary:
            attention_mask = attention_mask < 0.5
        else:
            attention_mask = None

        return loss_mask, position_ids, attention_mask
