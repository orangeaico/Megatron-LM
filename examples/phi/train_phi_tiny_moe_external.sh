#!/bin/bash

# Runs Phi-tiny-MoE-instruct model (simplified version)

PHASE_LOGGER=1
# optional for per-layer peaks
PHASE_LAYER_LOGGER=1

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NCCL_NVLS_ENABLE=0

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR-"localhost"}
MASTER_PORT=${MASTER_PORT-"6000"}
NNODES=${SLURM_NNODES-"1"}
NODE_RANK=${RANK-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=${1-"checkpoints/phi_tiny_moe_instruct"}
TOKENIZER_MODEL=${2-"MOCK"}
DATA_PATH=${3-"MOCK"}

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
  --distributed-timeout-minutes 60
  --tensor-model-parallel-size 2
  --pipeline-model-parallel-size 1
  --expert-model-parallel-size 1
  --context-parallel-size 1
  --expert-tensor-parallel-size 1
  --use-distributed-optimizer 
  --overlap-grad-reduce 
  --overlap-param-gather 
  --no-create-attention-mask-in-dataloader 

  # Training args
  --use-mcore-models 
  --sequence-parallel 
  --use-flash-attn 
  --disable-bias-linear 
  --micro-batch-size 1
  --global-batch-size 1
  --train-samples 2000
  --exit-duration-in-mins 230
  --manual-gc 
  --manual-gc-interval 5
  --cross-entropy-loss-fusion 
  --cross-entropy-fusion-impl te
  --enable-experimental 

  # Transformer Engine args
  --transformer-impl transformer_engine

  # Data args
  --mock-data
  --data-cache-path $DATA_CACHE_PATH
  --tokenizer-type NullTokenizer 
  --vocab-size 32064
  --split 99,1,0
  --no-mmap-bin-files 
  --no-create-attention-mask-in-dataloader
  --num-workers 1

  # Add network size args
  --untie-embeddings-and-output-weights 
  --position-embedding-type rope
  --rotary-percent 1.0
  --rotary-base 1000000
  --rotary-seq-len-interpolation-factor 1
  --normalization RMSNorm
  --swiglu 
  --norm-epsilon 1e-06
  --num-layers 8
  --hidden-size 4096
  --ffn-hidden-size 448
  --num-attention-heads 16
  --group-query-attention 
  --num-query-groups 4
  --kv-channels 128
  --qk-layernorm 
  --seq-length 2048
  --max-position-embeddings 40960
  --make-vocab-size-divisible-by 1187

  # Add regularization args
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --clip-grad 1.0
  --weight-decay 0.1

  # Add learning rate args
  --lr-decay-samples 255126953
  --lr-warmup-samples 162761
  --lr 1.2e-4
  --min-lr 1.2e-5
  --lr-decay-style cosine
  --adam-beta1 0.9
  --adam-beta2 0.95

  # Add MoE args
  --num-experts 16
  --moe-ffn-hidden-size 448
  --moe-router-load-balancing-type aux_loss
  --moe-router-topk 2
  --moe-grouped-gemm
  --moe-aux-loss-coeff 1e-3
  --moe-token-dispatcher-type alltoall
  --moe-permute-fusion 
  --moe-router-dtype fp32

  # Add validation args
  --eval-iters 32
  --eval-interval 500

  # Add checkpointing args
  --auto-detect-ckpt-format 
  --load $CHECKPOINT_PATH
  --save $CHECKPOINT_PATH
  --save-interval 500
  --dist-ckpt-strictness log_all

  # Add initialization args
  --init-method-std 0.02

  # Add logging args
  --log-timers-to-tensorboard 
  --log-memory-to-tensorboard 
  --log-num-zeros-in-grad 
  --log-params-norm 
  --log-validation-ppl-to-tensorboard 
  --log-throughput 
  --log-interval 1
  --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"

  # Add mixed precision args
    --bf16
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} 