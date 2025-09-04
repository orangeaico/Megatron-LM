#!/bin/bash

# Runs Phi-tiny-MoE-instruct model (simplified version)

PHASE_LOGGER=1
# optional for per-layer peaks:
PHASE_LAYER_LOGGER=1

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=${1:-"checkpoints/phi_tiny_moe_instruct"}
TOKENIZER_MODEL=${2:-"MOCK"}
DATA_PATH=${3:-"MOCK"}

# Data cache path (useful for both mock and real data)
DATA_CACHE_PATH="${PWD}/benchmark_cache_phi_tiny"
mkdir -p "$DATA_CACHE_PATH"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 32768
    --max-position-embeddings 40960
    --num-layers 4
    --hidden-size 4096
    --ffn-hidden-size 448
    --num-attention-heads 16  # Phi-tiny-MoE attention heads
    --group-query-attention
    --num-query-groups 4  # Same as num-attention-heads for Phi
    --kv-channels 128
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 10000  # Standard RoPE base for Phi
)

MOE_ARGS=(
    --num-experts 16 # Phi-tiny-MoE has 16 experts
    --moe-ffn-hidden-size 448
    --moe-router-topk 2  # num_experts_per_tok
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0.0  # router_aux_loss_coef from config
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-permute-fusion
)

DATA_ARGS=()
if [[ "$TOKENIZER_MODEL" == "MOCK" ]] || [[ "$DATA_PATH" == "MOCK" ]] || [[ -z "$TOKENIZER_MODEL" ]]; then
    DATA_ARGS+=(
        "--mock-data"
        "--tokenizer-type NullTokenizer"
        "--vocab-size 32064" 
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--tiktoken-pattern v2" 
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
    )
else
    # Settings for real data
    DATA_ARGS+=(
        "--data-path $DATA_PATH"
        "--tokenizer-type HuggingFaceTokenizer" 
        "--tokenizer-model $TOKENIZER_MODEL"
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
        # Note: --vocab-size might be inferred by HuggingFaceTokenizer or might need to be explicit.
        "--vocab-size 32064"
    )
fi

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1
    --lr 3e-4  # Higher LR for smaller model
    --train-samples 100
    --lr-decay-samples 50
    --lr-warmup-samples 10
    --lr-decay-style cosine
    --min-lr 3.0e-5  # Adjusted min LR
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --exit-duration-in-mins 235
    --use-flash-attn
    --bf16
    --use-precision-aware-optimizer
    --main-params-dtype fp16
    --main-grads-dtype bf16
    --exp-avg-dtype fp16
    --exp-avg-sq-dtype fp16
    --grad-reduce-in-bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --context-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --overlap-grad-reduce 
    --overlap-param-gather
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --no-load-optim \
    --no-load-rng
)

LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 32
    --eval-interval 100
    --save-interval 1000
    --log-throughput
    --profile
    --profile-step-start 4
    --profile-step-end 6
    --ckpt-format torch_dist 
    --distributed-timeout-minutes 60
    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH" 
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
    --no-load-optim
    --no-load-rng
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Phi-tiny-MoE"}
        --wandb-exp-name ${WANDB_NAME:-"Phi_tiny_MoE_instruct"}
    )
fi


torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}