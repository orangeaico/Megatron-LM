#!/usr/bin/env bash
set -euo pipefail

# The input to the script is the timestamp of the current run
TIMESTAMP=$1

MODEL_NAME=Qwen3-Coder-30B-A3B-Instruct
TORCH_CHECKPOINTS_DIR_PATH="/workspace/data/himanshu/output/$TIMESTAMP/$MODEL_NAME/checkpoints/"

echo "Torch model checkpoints directory: $TORCH_CHECKPOINTS_DIR_PATH"

cd $TORCH_CHECKPOINTS_DIR_PATH
# Cat the file in $TORCH_MODEL_PATH/latest_checkpointed_iteration.txt
RELEVANT_ITERATION=$(cat latest_checkpointed_iteration.txt)

# Format the RELEVANT_ITERATION 7 digits and add 0s at the beginning to make it 7 digits
RELEVANT_ITERATION=$(printf "%07d" $RELEVANT_ITERATION)

MODEL_DIR="$TORCH_CHECKPOINTS_DIR_PATH/iter_$RELEVANT_ITERATION"

echo "Reading torch model from: $MODEL_DIR"
# Rename the model subdirectories in the format expected by Pai Megatron converter
cd $MODEL_DIR

# Check if the model subdirectories are already in the format expected by Pai Megatron converter
if [ -d mp_rank_00_000_000 ]; then
    echo "The model subdirectories are already in the format expected by Pai Megatron converter"
else
    echo "Renaming the torch model subdirectories mp_rank_* in the format expected by Pai Megatron converter"
    mv mp_rank_00_000 mp_rank_00_000_000
    mv mp_rank_01_001 mp_rank_01_000_001
    mv mp_rank_02_002 mp_rank_02_000_002
    mv mp_rank_03_003 mp_rank_03_000_003
fi

cd /workspace
# Clone the Pai Megatron Patch repo if it doesn't exist
if [ ! -d Pai-Megatron-Patch ]; then
    git clone --recurse-submodules https://github.com/orangeaico/Pai-Megatron-Patch.git
fi
cd Pai-Megatron-Patch
git switch cpu_conversion
cd toolkits/model_checkpoints_convertor/qwen

# Convert the torch model to HF
echo "Converting the torch model to HF and saving to: /workspace/data/himanshu/output/$TIMESTAMP/$MODEL_NAME/conversion/qwen3_30b_a3b_hf/"

bash hf2mcore_qwen3_convertor.sh A3B /workspace/data/himanshu/output/$TIMESTAMP/$MODEL_NAME/checkpoints /workspace/data/himanshu/output/$TIMESTAMP/$MODEL_NAME/conversion/qwen3_30b_a3b_hf/ 4 1 1 4 bf16 true /workspace/data/mega-models/Qwen3-Coder-30B-A3B-Instruct_torch_tp4_ep4

# Rename the model subdirectories back to original names
echo "Renaming the torch model subdirectories back to original names"
cd $MODEL_DIR
mv mp_rank_00_000_000 mp_rank_00_000
mv mp_rank_01_000_001 mp_rank_01_001
mv mp_rank_02_000_002 mp_rank_02_002
mv mp_rank_03_000_003 mp_rank_03_003

# Copy the HF model to gdrive
echo "Copying the HF model to gdrive"
rclone copy /workspace/data/himanshu/output/$TIMESTAMP/$MODEL_NAME/conversion/qwen3_30b_a3b_hf/ gdrive:megatron_dir/himanshu/output/$TIMESTAMP/$MODEL_NAME/conversion/qwen3_30b_a3b_hf/ --progress

# Copy the logs as well
echo "Copying the training logs to gdrive"
rclone copy /workspace/data/himanshu/output/$TIMESTAMP/$MODEL_NAME/logs/ gdrive:megatron_dir/himanshu/output/$TIMESTAMP/$MODEL_NAME/logs/ --progress 

echo "All Done!"