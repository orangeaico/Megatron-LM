#!/usr/bin/env bash
set -euo pipefail
SETUP_DATA=1

mkdir -p "${DATA_DIRECTORY:-/workspace/}" /workspace/data "$HOME/.config/rclone"
cd /workspace

if [ "$SETUP_DATA" -eq 1 ]; then
  apt-get install -y rclone
  cat >"$HOME/.config/rclone/rclone.conf"<<'EOF'
[gdrive]
type = drive
scope = drive
client_id = 983615320622-9vfjc78upb9igrcf54i6dvb4cvecfpm3.apps.googleusercontent.com
client_secret = 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
token = {}
EOF
  chmod 600 "$HOME/.config/rclone/rclone.conf"

  ( cd /workspace/data && rclone copy -P --transfers 32 --checkers 64 --fast-list --buffer-size 128M \
      gdrive:"megatron_dir/data/sft/" data/sft/ ) &
fi

( cd /workspace && { [ -d Megatron-LM ] || git clone https://github.com/orangeaico/Megatron-LM.git; } && \
  cd Megatron-LM && git checkout moe_experiments && \
  pip install unsloth transformers==4.56.2 && pip install --no-deps trl==0.22.2) &
  # pip install accelerate deepspeed datasets transformers trl peft bitsandbytes tensorboardX flash-attn) &

for pid in $(jobs -p); do wait "$pid" || exit 1; done

cd /workspace && git clone https://github.com/woct0rdho/transformers-qwen3-moe-fused.git
cp /workspace/Megatron-LM/examples/lora/fused_train_30b_a3b_unsloth.py /workspace/transformers-qwen3-moe-fused/
cp /workspace/Megatron-LM/examples/lora/conversion.py /workspace/transformers-qwen3-moe-fused/

echo "Downloading model from Huggingface..."
hf download Qwen/Qwen3-Coder-30B-A3B-Instruct --local-dir /workspace/data/Qwen3-Coder-30B-A3B-Instruct/

echo "Converting model to fused format..."
cd /workspace/transformers-qwen3-moe-fused && python conversion.py

cd /workspace/Megatron-LM && echo "All Done!"
