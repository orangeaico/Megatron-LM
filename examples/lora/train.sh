
export TOKENIZERS_PARALLELISM=false

accelerate launch --config_file /home/himanshu/megatron_dir/Megatron-LM/examples/lora/accelerate_config.yaml \
/home/himanshu/megatron_dir/Megatron-LM/examples/lora/train_qwen3coder_lora.py \
  --model_name /home/shared/megatron_dir/hf_models/Qwen3-1.7B \
  --train_file /home/shared/megatron_dir/data/sft/train_loc_xarray_256_samples_16k_max_tokens/LocFileFuncLine_train_shuf.jsonl \
  --eval_file /home/shared/megatron_dir/data/sft/train_loc_xarray_256_samples_16k_max_tokens/LocFileFuncLine_val_shuf.jsonl \
  --output_dir /home/shared/megatron_dir/output/himanshu/qwen3coder30b-lora \
  --max_seq_len 12288 \
  --per_device_train_bs 1 \
  --grad_accum 8 \
  --epochs 4 \
  --lr 1e-4 \
  --use_qlora \
  --bf16 \
  --gradient_checkpointing \