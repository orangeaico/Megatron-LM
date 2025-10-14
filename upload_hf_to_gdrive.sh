#!/bin/bash
set -euo pipefail

TIMESTAMP=$1
MODEL_NAME="Qwen3-Coder-30B-A3B-Instruct"
BASE_CONVERSION_DIR="/workspace/data/himanshu/output/$TIMESTAMP/$MODEL_NAME/conversion/"

echo "Searching for *_fp8 directories inside: $BASE_CONVERSION_DIR"
echo "Upload timestamp: $TIMESTAMP"

# Loop through all *_fp8 directories
find "$BASE_CONVERSION_DIR" -type d -name "*_fp8" | while read -r FP8_HF_MODEL_PATH; do
    FP8_HF_MODEL_NAME=$(basename "$FP8_HF_MODEL_PATH")
    DESTINATION="gdrive:megatron_dir/himanshu/output/${TIMESTAMP}/${MODEL_NAME}/conversion/${FP8_HF_MODEL_NAME}/"

    echo "Uploading: $FP8_HF_MODEL_PATH"
    echo "Destination: $DESTINATION"
    echo "---------------------------------------------"

    # Perform the upload
    rclone copy "$FP8_HF_MODEL_PATH" "$DESTINATION" --progress

    echo "✅ Upload complete for $FP8_HF_MODEL_NAME"
    echo
done

echo "🎉 All FP8 directories have been uploaded successfully!"