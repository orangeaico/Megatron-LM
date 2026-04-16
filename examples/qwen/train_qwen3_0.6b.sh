#!/bin/bash

#export LOG_LEVEL=${LOG_LEVEL:-INFO}
#export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-19}
#export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
#export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}
#export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-2097152}
#export NCCL_AVOID_RECORD_STREAMS=${NCCL_AVOID_RECORD_STREAMS:-1}

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NCCL_NVLS_ENABLE=0

# Set these to 1 to enable torch profiling and nvidia nsys profiling
ENABLE_PROFILING=0
ENABLE_NSYS_PROFILING=0

# CRITICAL - DOUBLE CHECK THIS VALUE
TRAINING_MODE="sft" # set from mock, cpt, sft or distillation

MODEL_NAME="qwen3_0.6b"

TIMESTAMP=$(date +"%Y_%m_%d_%H_%M_%S")
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/shramana/output}"

LOAD_CHECKPOINT_PATH="${LOAD_CHECKPOINT_PATH:-$DATA_ROOT/mega-models/Qwen3-0.6B}"
TOKENIZER_ARG="${TOKENIZER_ARG:-$DATA_ROOT/hf_models/Qwen3-0.6B}"

echo "Training mode: $TRAINING_MODE"

if [[ "$TRAINING_MODE" == "cpt" ]]; then
    TRAIN_DATA_PATH="$DATA_ROOT/data/sft/pretraining/xarray_ctx4096_diff_16k_combined_text_document"
    VALID_DATA_PATH="$DATA_ROOT/data/sft/pretraining/xarray_validation_ctx4096_text_document"
    TEST_DATA_PATH=$VALID_DATA_PATH

elif [[ "$TRAINING_MODE" == "sft" ]]; then
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$DATA_ROOT/data/sft/gsm8k/training_pass_rate_05.jsonl}"
    VALID_DATA_PATH="${VALID_DATA_PATH:-$DATA_ROOT/data/sft/gsm8k/eval_set.jsonl}"
    TEST_DATA_PATH="${TEST_DATA_PATH:-$VALID_DATA_PATH}"

elif [[ "$TRAINING_MODE" == "distillation" ]]; then
    TRAIN_DATA_PATH="$DATA_ROOT/data/distillation/qwen_480b_swe_bench/"
    VALID_DATA_PATH="$DATA_ROOT/data/distillation/qwen_480b_swe_bench_excluded/"
    TEST_DATA_PATH=$VALID_DATA_PATH

elif [[ "$TRAINING_MODE" == "mock" ]]; then
    TRAIN_DATA_PATH="MOCK"
else
    echo "Training mode should be one of mock, cpt, sft or distillation. Invalid training mode: $TRAINING_MODE"
    exit 1
fi

BASE_OUTPUT_DIR="$OUTPUT_ROOT/$TIMESTAMP"
SAVE_CHECKPOINT_PATH="$BASE_OUTPUT_DIR/$MODEL_NAME/checkpoints"
# Data cache path (useful for both mock and real data)
DATA_CACHE_PATH="$BASE_OUTPUT_DIR/$MODEL_NAME/benchmark_cache"
TENSORBOARD_LOGS_PATH="$BASE_OUTPUT_DIR/$MODEL_NAME/tensorboard_logs"
MEMORY_SNAPSHOT_PATH="$BASE_OUTPUT_DIR/$MODEL_NAME/memory_snapshots/memory_snapshot.pickle"
LOG_DIR_PATH="$BASE_OUTPUT_DIR/$MODEL_NAME/logs"
CONVERSION_DIR_PATH="$BASE_OUTPUT_DIR/$MODEL_NAME/conversion"

echo "Timestamp: $TIMESTAMP"
echo "Load checkpoint path: $LOAD_CHECKPOINT_PATH"
echo "Tokenizer path: $TOKENIZER_ARG"
echo "TRAIN DATA PATH: $TRAIN_DATA_PATH"
echo "BASE OUTPUT DIR: $BASE_OUTPUT_DIR"

export WANDB_API_KEY=${WANDB_API_KEY:-}

# Create directories if they don't exist
mkdir -p "$(dirname "$SAVE_CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"
mkdir -p "$(dirname "$MEMORY_SNAPSHOT_PATH")"
mkdir -p "$DATA_CACHE_PATH"
mkdir -p "$LOG_DIR_PATH"
mkdir -p "$CONVERSION_DIR_PATH"

# Distributed training setup
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Path to the pretrain_gpt.py script, assuming this script is run from the root of the Megatron-LM repository
PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

# Fixed model and training parameters for Qwen3-0.6B
TP_SIZE=${TP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
NUM_LAYERS=28
DTYPE="bf16"
SEQ_LENGTH=${SEQ_LENGTH:-4096}
MAX_POSITION_EMBEDDINGS=40960
ATTENTION_DROPOUT=${ATTENTION_DROPOUT:-0.0}
HIDDEN_DROPOUT=${HIDDEN_DROPOUT:-0.0}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-auto}
NUM_EPOCHS=${NUM_EPOCHS:-4}

if [[ "$TRAINING_MODE" == "sft" ]]; then
    TRAIN_DATASET_SIZE=${TRAIN_DATASET_SIZE:-$(wc -l < "$TRAIN_DATA_PATH")}
    TRAIN_SAMPLES=${TRAIN_SAMPLES:-$((TRAIN_DATASET_SIZE * NUM_EPOCHS))}
    LR_DECAY_SAMPLES=${LR_DECAY_SAMPLES:-$TRAIN_SAMPLES}
    STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-$(((TRAIN_DATASET_SIZE + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE))}
    SAVE_INTERVAL=${SAVE_INTERVAL:-$STEPS_PER_EPOCH}
    EVAL_INTERVAL=${EVAL_INTERVAL:-$STEPS_PER_EPOCH}
else
    TRAIN_SAMPLES=${TRAIN_SAMPLES:-80}
    LR_DECAY_SAMPLES=${LR_DECAY_SAMPLES:-$TRAIN_SAMPLES}
    SAVE_INTERVAL=${SAVE_INTERVAL:-48}
    EVAL_INTERVAL=${EVAL_INTERVAL:-24}
fi

LR_WARMUP_SAMPLES=${LR_WARMUP_SAMPLES:-70}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers $NUM_LAYERS
    --seq-length $SEQ_LENGTH
    --hidden-size 1024
    --ffn-hidden-size 3072
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --qk-layernorm
    --normalization RMSNorm
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --make-vocab-size-divisible-by 1187
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --rotary-seq-len-interpolation-factor 1
    --swiglu
    --norm-epsilon 1e-06
    --init-method-std 0.02
    --disable-bias-linear
)

TRAINING_ARGS=(
    --optimizer adam
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples $TRAIN_SAMPLES
    --lr-decay-samples $LR_DECAY_SAMPLES

    # Learning rate args
    --lr-warmup-samples $LR_WARMUP_SAMPLES
    --lr 1.0e-5
    --min-lr 3.0e-6
    --lr-decay-style cosine
    --adam-beta1 0.9
    --adam-beta2 0.95

    # Regularization args
    --attention-dropout $ATTENTION_DROPOUT
    --hidden-dropout $HIDDEN_DROPOUT
    --clip-grad 1.0
    --weight-decay 0.0
 
    # Memory cleanup args
    --manual-gc
    --manual-gc-interval 5  

    # Computation optimisation and recomputation args
    --transformer-impl transformer_engine
    --enable-experimental
    --attention-backend $ATTENTION_BACKEND
    --use-flash-attn
    --fused-linear-cross-entropy
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    # --no-gradient-accumulation-fusion

    # data type arguments
    --bf16
    --use-distributed-optimizer
    --use-precision-aware-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --main-params-dtype fp16
    --main-grads-dtype bf16
    --grad-reduce-in-bf16
    --exp-avg-dtype fp16
    --exp-avg-sq-dtype fp16
    # --optimizer-cpu-offload
    # --optimizer-offload-fraction 1.0
    # --overlap-cpu-optimizer-d2h-h2d
    # --use-torch-optimizer-for-cpu-offload
)

# Conditional arguments based on DTYPE (FP8)
DTYPE_ARGS=()
if [[ "$DTYPE" == "fp8" ]]; then
    DTYPE_ARGS+=(
        "--fp8-format hybrid"
        "--fp8-amax-history-len 1024"
        "--fp8-amax-compute-algo max"
        "--fp8-param-gather"
    )
fi

# Model parallelism arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size $CP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
)
if [[ "${SEQUENCE_PARALLEL:-1}" == "1" ]]; then
    MODEL_PARALLEL_ARGS+=(--sequence-parallel)
fi

# Data arguments (conditional for mock vs real data)
DATA_ARGS_LIST=(
    "--vocab-size 151936"
    "--no-mmap-bin-files"
    # "--data-cache-path ${DATA_CACHE_PATH}"
    # "--no-check-for-nan-in-loss-and-grad"
)
if [[ "$TRAINING_MODE" == "mock" ]]; then
    DATA_ARGS_LIST+=(
        "--mock-data"
        "--tokenizer-type NullTokenizer"        
        "--tiktoken-pattern v2" 
        "--split '99,1,0'"
        "--num-workers 1"
        "--no-create-attention-mask-in-dataloader"                      
    )
elif [[ "$TRAINING_MODE" == "cpt" ]]; then
    # Settings for real data
    DATA_ARGS_LIST+=(
        "--train-data-path $TRAIN_DATA_PATH"
        "--valid-data-path $VALID_DATA_PATH"
        "--test-data-path $TEST_DATA_PATH"
        "--tokenizer-type HuggingFaceTokenizer" 
        "--tokenizer-model $TOKENIZER_ARG"
        "--num-workers 1"
        "--no-create-attention-mask-in-dataloader"
        # "--trsft"
        # "--trsft-alpha 0.05"
        # "--reset-position-ids"
        # "--reset-attention-mask"
        # "--eod-mask-loss"        
    )
elif [[ "$TRAINING_MODE" == "sft" ]]; then
    # Settings for real data
    DATA_ARGS_LIST+=(
        "--train-data-path $TRAIN_DATA_PATH"
        "--valid-data-path $VALID_DATA_PATH"
        "--test-data-path $TEST_DATA_PATH"
        # "--data-path $TRAIN_DATA_PATH"
        # "--split '95,5,0'"  
        "--tokenizer-type HuggingFaceTokenizer" 
        "--tokenizer-model $TOKENIZER_ARG"
        "--sft"
        "--num-workers 1"
        "--no-create-attention-mask-in-dataloader"
        "--trsft"
        "--trsft-alpha 0.05"
        # "--weighted-loss"
        "--variable-seq-lengths"                
        "--moe-token-dispatcher-type alltoall" # This needs to be set for variable seq lengths

        # "--reset-position-ids"
        # "--reset-attention-mask"
        # "--eod-mask-loss"        
    )
elif [[ "$TRAINING_MODE" == "distillation" ]]; then
    # Settings for real data
    DATA_ARGS_LIST+=(        
        "--train-data-path $TRAIN_DATA_PATH"
        "--valid-data-path $VALID_DATA_PATH"
        "--test-data-path $TEST_DATA_PATH"        
        "--tokenizer-type HuggingFaceTokenizer" 
        "--tokenizer-model $TOKENIZER_ARG"
        "--sft"
        "--num-workers 1"
        "--no-create-attention-mask-in-dataloader"         
        "--distillation-loss"
        "--distillation-temperature 1.0"
        "--distillation-loss-alpha 0"     
        "--variable-seq-lengths"                
        "--moe-token-dispatcher-type alltoall" 
    )
else
    echo "Training mode should be one of mock, cpt, sft or distillation. Invalid training mode: $TRAINING_MODE"
    exit 1
fi

CHECKPOINT_ARGS=(
    --finetune
    --auto-detect-ckpt-format
    --dist-ckpt-strictness raise_unexpected
    --distributed-timeout-minutes 60
    --load "$LOAD_CHECKPOINT_PATH"
    --save "$SAVE_CHECKPOINT_PATH"
    --no-save-optim
    --no-save-rng
    --no-load-rng
    --no-load-optim
    --save-interval $SAVE_INTERVAL
    --exit-on-missing-checkpoint
)

EVAL_AND_LOGGING_ARGS=(
    --eval-iters 1
    --eval-interval $EVAL_INTERVAL
    # --full-validation
    --log-interval 1
    --log-throughput
    --log-num-zeros-in-grad
    --log-params-norm
)

if [[ "$ENABLE_PROFILING" == 1 ]]; then
    EVAL_AND_LOGGING_ARGS+=(
        --profile
        --profile-step-start 2
        --profile-step-end 3
        --profile-ranks 0
        --use-pytorch-profiler
        --log-timers-to-tensorboard
        --log-validation-ppl-to-tensorboard
        --log-memory-to-tensorboard
        --record-memory-history
        --memory-snapshot-path "$MEMORY_SNAPSHOT_PATH"
        --timing-log-level 2
        --logging-level 10
        --timing-log-option all
        --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
        # --dump-model-params-to-pickle
    )
fi

if [ -n "${WANDB_API_KEY}" ]; then
    EVAL_AND_LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-$MODEL_NAME}
        --wandb-exp-name ${WANDB_NAME:-$MODEL_NAME}
    )
fi

if [[ "$ENABLE_NSYS_PROFILING" == 1 ]]; then
    NSYS_PROFILE_COMMAND="nsys profile -o $LOG_DIR_PATH/nsys_run -t cuda,nvtx,osrt --sample=none --cpuctxsw=none"
else
    NSYS_PROFILE_COMMAND=""
fi

# Ensure pretrain_gpt.py is found
if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    echo "Please ensure you are running this script from the root of the Megatron-LM repository, and pretrain_gpt.py is present."
    exit 1
fi

# Run the training command
$NSYS_PROFILE_COMMAND torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${CHECKPOINT_ARGS[@]}    

set +x
