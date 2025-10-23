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

cd /workspace/Megatron-LM && echo "All Done!"
