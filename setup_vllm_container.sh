#!/usr/bin/env bash
set -euo pipefail

SETUP_DATA=1

env >> /etc/environment
mkdir -p ${DATA_DIRECTORY:-/workspace/}

cd /workspace

mkdir -p /workspace/data/

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
token = {"access_token":"ya29.a0AQQ_BDRFpxdfVvEThJD5xALTDJ5XciVW36HYBE0F8N0tA-gujurULRicrbBBMmQDtddyj2sLKLEaQWsu9skyopTDKn8lkEc4jbF7RxkzQ8IgrSFx6921wtPkSLYk1dA01v6lhG_s7VRXz04fUCQqmL7C8jJTUzrBhDQDkwob_MMWmpzZfKE0Ft9LOHdHJbBIdwh6BEEaCgYKAXwSARcSFQHGX2MiaO2wiHfWH0kDJghW3omc3g0206","token_type":"Bearer","refresh_token":"1//0g9XJCyR9hUR8CgYIARAAGBASNwF-L9Ir2ebGwJ6CNPVfSveAPJfmQI6bhgebJi6a5ZA_pybM-pYrOHqyL2PPskYltN3WtzI3NHM","expiry":"2025-10-13T18:19:07.314428+05:30","expires_in":3599}
EOF

chmod 600 "$CONFIG_PATH"

echo "✅ rclone.conf written to $CONFIG_PATH"

# Force a token refresh immediately and verify access
rclone about gdrive: -vv

echo "🎉 Google Drive remote [gdrive] is ready!"
fi

pip install vllm==0.10.1.1

cd /workspace/data/

echo "All Done!"