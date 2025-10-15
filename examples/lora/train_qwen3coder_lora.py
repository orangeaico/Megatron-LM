#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--eval_file", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="qwen3coder30b-lora")
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--per_device_train_bs", type=int, default=1)
    p.add_argument("--per_device_eval_bs", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=1)              # log every step
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--use_qlora", action="store_true", default=True)
    p.add_argument("--no-use_qlora", dest="use_qlora", action="store_false")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--local_files_only", action="store_true", default=True)
    return p.parse_args()


def load_chat_jsonl(path):
    return load_dataset("json", data_files=path, split="train")


# ---- Subclass to count tokens per micro-step

class LoggingSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._accum_total_tokens = 0
        self._accum_labeled_tokens = 0
        self._accum_samples = 0
        self._step_start_time = None

    # NOTE: accept num_items_in_batch and pass it to super()
    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        if self._step_start_time is None:
            self._step_start_time = time.perf_counter()

        # count tokens/samples for this micro-step
        with torch.no_grad():
            attn = inputs.get("attention_mask")
            total_tokens = int(attn.sum().item()) if attn is not None else 0
            labels = inputs.get("labels")
            labeled_tokens = int((labels != -100).sum().item()) if labels is not None else 0
            input_ids = inputs.get("input_ids")
            micro_samples = int(input_ids.size(0)) if input_ids is not None else 0

        loss = super().training_step(
            model, inputs, num_items_in_batch=num_items_in_batch
        )

        self._accum_total_tokens += total_tokens
        self._accum_labeled_tokens += labeled_tokens
        self._accum_samples += micro_samples
        return loss


class ThroughputCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # In HF, callbacks have a .trainer attribute set when attached.
        trainer: LoggingSFTTrainer = getattr(self, "trainer", None)
        if trainer is None or trainer._step_start_time is None:
            return control

        elapsed = time.perf_counter() - trainer._step_start_time
        dev = trainer.model.device

        def red(x):
            t = torch.tensor([float(x)], device=dev)
            if dist.is_initialized():
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return t.item()

        tot = red(trainer._accum_total_tokens)
        lab = red(trainer._accum_labeled_tokens)
        sam = red(trainer._accum_samples)

        tps = (tot / elapsed) if elapsed > 0 else 0.0
        sps = (sam / elapsed) if elapsed > 0 else 0.0
        try:
            mem_gb = torch.cuda.max_memory_allocated(dev) / 1e9
        except Exception:
            mem_gb = 0.0

        trainer.log({
            "throughput/step_time_sec": elapsed,
            "throughput/tokens_per_sec": tps,
            "throughput/samples_per_sec": sps,
            "counts/total_tokens_global": tot,
            "counts/labeled_tokens_global": lab,
            "pct/labeled_tokens": (lab / max(tot, 1.0)),
            "gpu/max_mem_allocated_gb": mem_gb,
        })

        # reset for next optimizer step
        trainer._accum_total_tokens = 0
        trainer._accum_labeled_tokens = 0
        trainer._accum_samples = 0
        trainer._step_start_time = None
        return control



def main():
    args = get_args()
    torch.backends.cuda.matmul.allow_tf32 = True

    tok = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=True, local_files_only=args.local_files_only
    )
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant_cfg = None
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    if args.use_qlora:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        quantization_config=quant_cfg,
        torch_dtype=dtype if quant_cfg is None else None,
        device_map=None,
    )

    # align PAD/BOS/EOS explicitly (silences warnings)
    model.config.pad_token_id = tok.pad_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.config.bos_token_id = tok.bos_token_id
    model.generation_config.bos_token_id = tok.bos_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.generation_config.eos_token_id = tok.eos_token_id

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )

    # ---- Build prompt–completion dataset
    def to_prompt_completion(example):
        msgs = example["messages"]
        last_ass_idx = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=-1)
        if last_ass_idx < 0:
            return {"prompt": None, "completion": None}
        prompt_msgs = msgs[:last_ass_idx]
        completion = msgs[last_ass_idx].get("content", "")
        prompt = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "completion": completion}

    train_raw = load_chat_jsonl(args.train_file)
    train_pc = train_raw.map(to_prompt_completion, remove_columns=train_raw.column_names)
    train_pc = train_pc.filter(lambda ex: ex["prompt"] is not None and ex["completion"] is not None)

    eval_pc = None
    if args.eval_file:
        eval_raw = load_chat_jsonl(args.eval_file)
        eval_pc = eval_raw.map(to_prompt_completion, remove_columns=eval_raw.column_names)
        eval_pc = eval_pc.filter(lambda ex: ex["prompt"] is not None and ex["completion"] is not None)

    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_bs,
        per_device_eval_batch_size=args.per_device_eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,

        # <- make logs every optimizer step and include step 0
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=args.logging_steps,

        save_steps=args.save_steps,
        eval_strategy=("steps" if eval_pc is not None else "no"),
        eval_steps=(args.eval_steps if eval_pc is not None else None),
        bf16=args.bf16,
        fp16=not args.bf16,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
        report_to=["tensorboard"],   # add "wandb" here if you use it
        save_total_limit=3,

        # TRL-specific
        max_length=args.max_seq_len,
        packing=False,
        completion_only_loss=True,
        dataset_text_field=None,
    )

    trainer = LoggingSFTTrainer(
        model=model,
        processing_class=tok,
        peft_config=lora_cfg,
        train_dataset=train_pc,
        eval_dataset=eval_pc,
        args=cfg,
    )
    trainer.add_callback(ThroughputCallback())

    trainer.train()
    trainer.save_model()
    trainer.save_state()

    is_master = (not dist.is_initialized()) or (dist.get_rank() == 0)
    if is_master:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "run_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
