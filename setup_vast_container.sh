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
# SETUP_FA3=1 bash /workspace/Megatron-LM/setup_megatron_container.sh
mkdir -p /workspace/data/

echo "Megatron-LM setup complete without data setup!"

if [ "$SETUP_DATA" -eq 1 ]; then
echo "Setting up rclone and copying data and models from gdrive.."

echo "Installing rclone.."
apt-get update -y
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
token = {"access_token":"ya29.a0AQQ_BDS1Lp45dltHhflXU5L-CdoyRPl9Wq354M5BrKpzMWSDbWfm_fbuXMXbw-ec15emuRSjDmbcq2cWWGq-Pt9AWWrgG3Eiyd-iEfi4JW4IIqiYWU4WapZYRrk4LrdG4OkdZ5Y-xwzwZfTPhnbgUdyNOU-POpk3qd5T_0Jaot-qBF5T5IIVqBYbDWRwohFJM7WEw1UaCgYKASwSARcSFQHGX2MikdRGYxo0ZjhQDDZAqjfGsw0206","token_type":"Bearer","refresh_token":"1//0gPkbZqTctSIhCgYIARAAGBASNwF-L9IrarxV3inSr8UC3Q5K9DL6b0fwN7Xf2yHEIYuRURj-dxOg-TzbIiMyygyR-F_Xd8cn5ks","expiry":"2025-10-08T15:14:08.731401+05:30","expires_in":3599}
EOF

chmod 600 "$CONFIG_PATH"

echo "✅ rclone.conf written to $CONFIG_PATH"

# Force a token refresh immediately and verify access
rclone about gdrive: -vv

echo "🎉 Google Drive remote [gdrive] is ready!"

echo "Copying Files from gdrive to /workspace/data/"

cd /workspace/data/

echo "Copying data to /workspace/data/"
rclone copy -P gdrive:"megatron_dir/data/" data/

echo "Copying model to /workspace/data/mega-models/"
rclone copy -P gdrive:"megatron_dir/mega-models/Qwen3-Coder-30B-A3B-Instruct-torch_dist" mega-models/Qwen3-Coder-30B-A3B-Instruct-torch_dist

cd /workspace/Megatron-LM/
echo "Data copying complete!"
echo "Changed working directory to /workspace/Megatron-LM/"
fi

echo "All Done!"