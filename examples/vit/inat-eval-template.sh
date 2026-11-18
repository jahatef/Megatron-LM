#!/bin/bash
# Eval-only config: loads a trained checkpoint, runs validation once, exits.
# No training, no optimizer/RNG restore, no checkpoint saving.

set -o pipefail   # keep torchrun's exit code through the tee below

export GPUS_PER_NODE=${GPUS_PER_NODE:-2}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NUM_NODES=$SLURM_JOB_NUM_NODES
export NODE_RANK=${SLURM_NODEID:-0}
export WORLD_SIZE=$(($GPUS_PER_NODE * $NUM_NODES))

TRAIN_IMG_SIZE={{TRAIN_IMG_SIZE}}
EVAL_IMG_SIZE={{EVAL_IMG_SIZE}}
ROPE_TYPE="{{ROPE_TYPE}}"
ROPE_BASE_TAG="{{ROPE_BASE_TAG}}"
ROPE_FRACTION="{{ROPE_FRACTION}}"
LOAD_DIR="{{LOAD_DIR}}"
EXP_NAME="{{EXP_NAME}}"
EVAL_ROOT="/home/hatef.4/megatron/Megatron-LM/evals"

mkdir -p "$EVAL_ROOT/logs" "$EVAL_ROOT/wandb" "$EVAL_ROOT/tensorboard/$EXP_NAME"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --node_rank $NODE_RANK
)

GPT_MODEL_ARGS=(
    --num-layers 24
    --hidden-size 1024
    --num-attention-heads 16
    --seq-length 196
    --max-position-embeddings 196
    --position-embedding-type rope
    --vit-rope-impl {{ROPE_TYPE}}
    {{ROPE_BASE_LINE}}
    --vit-rotary-fraction {{ROPE_FRACTION}}
    --img-size $EVAL_IMG_SIZE
    --patch-dim 16
    --attention-backend flash
    --transformer-impl transformer_engine
)

# Training loop is skipped entirely; these only satisfy arg validation
# and size the dataset/scheduler.
TRAINING_ARGS=(
    --skip-train
    --micro-batch-size {{MICRO_BATCH}}
    --global-batch-size {{GLOBAL_BATCH}}
    --train-iters 1
    --lr 1.0e-4
    --min-lr 1.0e-6
    --lr-decay-style constant
    --weight-decay 0.2
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --bf16
)

LOAD_ARGS=(
    --load "$LOAD_DIR"
    --ckpt-step {{CKPT_STEP}}   # load this iteration, not latest_checkpointed_iteration.txt
    --finetune          # load weights only, reset iteration counter
    --no-load-optim
    --no-load-rng
    --ckpt-format torch
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path "/home/hatef.4/datasets/inat-500"
    --dataloader-type 'cyclic'
    --vocab-file "/home/hatef.4/neox/gpt-neox/data/gpt2-vocab.json"
    --merge-file "/home/hatef.4/neox/gpt-neox/data/gpt2-merges.txt"
    --split 80,20,0
    --num-classes 500
)

EVAL_AND_LOGGING_ARGS=(
    --eval-interval 1
    --eval-iters {{EVAL_ITERS}}
    --log-interval 1
    --tensorboard-dir "$EVAL_ROOT/tensorboard/$EXP_NAME"
    --timing-log-level 0
    --wandb-project rope-evals
    --wandb-exp-name "$EXP_NAME"
    --wandb-save-dir "$EVAL_ROOT/wandb"
)
export NCCL_DEBUG=WARN

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_vision.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOAD_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    2>&1 | tee "$EVAL_ROOT/logs/$EXP_NAME-node${NODE_RANK}.log"