export TOKENIZERS_PARALLELISM=false
mkdir -p /workspace/data/himanshu/output/

accelerate launch --config_file /workspace/Megatron-LM/examples/lora/accelerate_config_vast.yaml \
/workspace/transformers-qwen3-moe-fused/fused_train_30b_a3b_unsloth.py \
  --model_name bash99/Qwen3-30B-A3B-Instruct-2507-fused-bnb-4bit \
  --train_file /workspace/data/data/sft/train_data_sft_480b_375_swe_bench_shuf.jsonl \
  --eval_file /workspace/data/data/sft/train_data_sft_480b_375_swe_bench_shuf_eval.jsonl \
  --output_dir /workspace/data/himanshu/output/qwen3coder30b-lora \
  --max_seq_len 65000 \
  --per_device_train_bs 1 \
  --per_device_eval_bs 1 \
  --grad_accum 8 \
  --epochs 4 \
  --lr 1e-4 \
  --use_qlora \
  --bf16 \
  --gradient_checkpointing \
  --use_flash_attn \