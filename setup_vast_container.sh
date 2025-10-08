#!/usr/bin/env bash
set -euo pipefail

SETUP_DATA=1

env >> /etc/environment
mkdir -p ${DATA_DIRECTORY:-/workspace/}

cd /workspace

if [ ! -d Megatron-LM ]; then
  git clone https://github.com/orangeaico/Megatron-LM.git
fi

cd Megatron-LM/
git checkout moe_experiments
SETUP_FA3=1 bash /workspace/Megatron-LM/setup_megatron_container.sh
mkdir -p /workspace/data/

echo "Megatron-LM setup complete without data setup!"

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
token = {"access_token":"ya29.a0AQQ_BDQc3eRiMLbMhOhLx20QyfOwSeY1iBlCTY2sG7ox-7Drn3F_H_JGBOqw7__z_5KebyxKo6KHb1k16J4KfWNXZmFiwGmyzTtUYb-OT61lmeF6lG2mmx_o9wbEDf0jBDO7NjeLW2-WrvIQsZOD06nCz6v5ez4SCoq0chJcZTpNSrV0AGLq0KHc94uF4OCEOJARSQoaCgYKAQMSARcSFQHGX2MilmEFQecTGlDBnGElN5lOCA0206","token_type":"Bearer","refresh_token":"1//0gt3EUPKSqbWrCgYIARAAGBASNwF-L9Irwi-Axig2zM4UoNRQ5ZprohdBfFO8R3EIK6Nq-trkZxvCMFHiFLR7Od4BuqzzwFsiDG4","expiry":"2025-10-08T16:40:04.545814+05:30","expires_in":3599}
EOF

chmod 600 "$CONFIG_PATH"

echo "✅ rclone.conf written to $CONFIG_PATH"

# Force a token refresh immediately and verify access
rclone about gdrive: -vv

echo "🎉 Google Drive remote [gdrive] is ready!"

cd /workspace/data/

echo "Copying data to /workspace/data/"
rclone copy -P gdrive:"megatron_dir/data/" data/

echo "Copying model to /workspace/data/mega-models/"
rclone copy -P gdrive:"megatron_dir/mega-models/Qwen3-Coder-30B-A3B-Instruct-torch_dist" mega-models/Qwen3-Coder-30B-A3B-Instruct-torch_dist

cd /workspace/Megatron-LM/
echo "Data copying complete!"
fi

echo "All Done!"