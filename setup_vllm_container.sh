#!/usr/bin/env bash
set -euo pipefail

SETUP_DATA=1

env >> /etc/environment
mkdir -p ${DATA_DIRECTORY:-/workspace/}

cd /workspace

mkdir -p /workspace/data/
mkdir -p /workspace/data/models/
mkdir -p /workspace/repo_eval/quantization/

if [ "$SETUP_DATA" -eq 1 ]; then
echo "Installing rclone.."
# apt-get update -y
apt-get install -y rclone

REMOTE_NAME="gdrive"
CONFIG_PATH="$HOME/.config/rclone/rclone.conf"
mkdir -p "$(dirname "$CONFIG_PATH")"

cat > "$CONFIG_PATH" <<'EOF'
[gdrive]
type = drive
scope = drive
client_id = 983615320622-9vfjc78upb9igrcf54i6dvb4cvecfpm3.apps.googleusercontent.com
client_secret = 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
token = {}
EOF

chmod 600 "$CONFIG_PATH"

echo "✅ rclone.conf written to $CONFIG_PATH"

# Force a token refresh immediately and verify access
rclone about gdrive: -vv

echo "🎉 Google Drive remote [gdrive] is ready!"
fi

rclone copy -P gdrive:megatron_dir/code/quantize_qwen_moe.py /workspace/repo_eval/quantization/

# pip install vllm==0.10.1.1 accelerate transformers

# rclone copy -P --transfers 13 gdrive:megatron_dir/himanshu/output/2025_11_04_09_56_13/Qwen3-Coder-30B-A3B-Instruct/conversion/qwen3_30b_a3b_0000188_hf/ /workspace/data/models/cpt/qwen3_30b_a3b_0000188_hf
# python3 /workspace/repo_eval/quantization/quantize_qwen_moe.py --src /workspace/data/models/cpt/qwen3_30b_a3b_0000188_hf --dst /workspace/data/models/cpt/qwen3_30b_a3b_0000188_hf_fp8

cd /workspace/data/

echo "All Done!"