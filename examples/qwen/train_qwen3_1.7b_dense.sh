#!/bin/bash

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
#export LOG_LEVEL=${LOG_LEVEL:-INFO}
#export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-19}
#export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
#export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}
#export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-2097152}
#export NCCL_AVOID_RECORD_STREAMS=${NCCL_AVOID_RECORD_STREAMS:-1}

CHECKPOINT_PATH=${1:-"checkpoints/qwen3_1.7b_fp8"}
TENSORBOARD_LOGS_PATH=${2:-"tensorboard_logs/qwen3_1.7b_fp8"}
TENSORBOARD_LOGS_PATH_MEMORY=${2:-"tensorboard_logs/qwen3_1.7b_fp8/memory_snapshots"}
TOKENIZER_ARG=${3:-"MOCK"} # Path to tokenizer model, or "MOCK"
DATA_ARG=${4:-"MOCK"}     # Data prefix, or "MOCK"

# Create directories if they don't exist
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"

# Distributed training setup
GPUS_PER_NODE=2
NUM_NODES=1
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Path to the pretrain_gpt.py script, assuming this script is run from the root of the Megatron-LM repository
PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

# Fixed model and training parameters for Qwen3-1.7B
TP_SIZE=2    
CP_SIZE=1     
PP_SIZE=1     
MICRO_BATCH_SIZE=1 # Increased from 1 due to smaller model size
GLOBAL_BATCH_SIZE=1  # Increased from 128 for better efficiency with smaller model
NUM_LAYERS=28  # Qwen3-1.7B has 28 layers
DTYPE="bf16"
SEQ_LENGTH=8192
MAX_POSITION_EMBEDDINGS=40960  # Qwen3-1.7B supports up to 40960

# Data cache path (useful for both mock and real data)
DATA_CACHE_PATH="${PWD}/benchmark_cache_qwen3_1.7b_fp8"
mkdir -p "$DATA_CACHE_PATH"

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
    --hidden-size 2048  # Qwen3-1.7B hidden size
    --ffn-hidden-size 6144  # Qwen3-1.7B intermediate size
    --num-attention-heads 16  # Qwen3-1.7B attention heads
    --group-query-attention
    --num-query-groups 8  # Qwen3-1.7B KV heads
    --kv-channels 128  # 2048 / 16 = 128
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --position-embedding-type rope
    --rotary-base 1000000  # Same as Qwen3 rope_theta
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu  # Qwen3 uses silu activation, compatible with swiglu
    --init-method-std 0.02  # Standard initialization for smaller model
    --attention-backend fused
    --apply-layernorm-1p 
    --untie-embeddings-and-output-weights
    --disable-bias-linear 
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples 2000
    --lr-decay-samples 1000
    --lr-warmup-samples 100
    --lr 0.0003  # Higher learning rate for smaller model
    --min-lr 0.00003  # Adjusted min learning rate
    --decoupled-lr 8.0e-4  # Adjusted for smaller model
    --decoupled-min-lr 8.0e-5  # Adjusted for smaller model
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --cross-entropy-loss-fusion
    --calculate-per-token-loss 
    --manual-gc 
    --empty-unused-memory-level 1 
    --exit-duration-in-mins 235 
    --recompute-activations
    --recompute-granularity full
    --use-flash-attn
    --bf16
    --use-precision-aware-optimizer
    --main-params-dtype fp16
    --main-grads-dtype bf16
    --exp-avg-dtype fp16
    --exp-avg-sq-dtype fp16
    --grad-reduce-in-bf16
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
    # --pipeline-model-parallel-size $PP_SIZE # Not explicitly set in llama script options, assume 1 if not multi-node PP
    --sequence-parallel  # Always enable sequence parallelism with TP_SIZE=2
)

# Distributed Data Parallel (DDP) arguments
# From original script's ddp_args
DDP_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)
TRAINING_ARGS+=("${DDP_ARGS[@]}")


# Data arguments (conditional for mock vs real data)
DATA_ARGS_LIST=()
if [[ "$TOKENIZER_ARG" == "MOCK" ]] || [[ "$DATA_ARG" == "MOCK" ]] || [[ -z "$TOKENIZER_ARG" ]]; then
    DATA_ARGS_LIST+=(
        "--mock-data"
        "--tokenizer-type NullTokenizer"
        "--vocab-size 151936"  # Qwen3-1.7B vocab size
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--tiktoken-pattern v2" 
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
    )
else
    # Settings for real data
    DATA_ARGS_LIST+=(
        "--data-path $DATA_ARG"
        "--tokenizer-type HuggingFaceTokenizer" 
        "--tokenizer-model $TOKENIZER_ARG"
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
        # Note: --vocab-size might be inferred by HuggingFaceTokenizer or might need to be explicit.
        "--vocab-size 151936"  # Qwen3-1.7B vocab size
    )
fi

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 32
    --eval-interval 100
    --save-interval 1000
    --log-throughput
    --profile
    --profile-step-start 4
    --profile-step-end 6
    --profile-ranks 0
    --ckpt-format torch_dist 
    --distributed-timeout-minutes 60
    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH" 
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
    --use-pytorch-profiler
    --log-memory-to-tensorboard
)

# Ensure pretrain_gpt.py is found
if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    echo "Please ensure you are running this script from the root of the Megatron-LM repository, and pretrain_gpt.py is present."
    exit 1
fi

# Run the training command
torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

set +x
