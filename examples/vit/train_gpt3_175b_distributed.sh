#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.01-py3
export CHECKPOINT_PATH="/workspace/checkpoints" 
export TENSORBOARD_LOGS_PATH="/workspace/megatron-lm/logs" #<Specify path>
export VOCAB_FILE="/workspace/data/gpt2-vocab.json"
export MERGE_FILE="/workspace/data/gpt2-merges.txt"
export DATA_PATH="/workspace/dataset/"
export GPUS_PER_NODE=1
# Change for multinode config
export MASTER_ADDR=localhost
export MASTER_PORT=6000
export NUM_NODES=1
export NODE_RANK=0
export WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=$1 #<Specify path>
TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
DATA_PATH=$5 #<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 12
    --hidden-size 768
    --num-attention-heads 12
    --seq-length 196 
    --max-position-embeddings 196 
    --position-embedding-type rope
    --vit-rotary-base 100,150,200,250,300,350,400,450,500,550,600,650
    --img-size 224 \
    --patch-dim 16 \
    --attention-backend flash # Can use (flash/fused/unfused/local)
    --transformer-impl transformer_engine
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 1
    --train-iters 200 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 2 
    --recompute-granularity selective 
    --profile
    --profile-step-start 3
    --profile-step-end 8
    --use-pytorch-profiler
    --record-memory-history
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1 
	--pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 80,20,0
    --num-classes 4
)

EVAL_AND_LOGGING_ARGS=( 
    --log-interval 1
    --save-interval 10000 
    --eval-interval 10 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 20
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --log-throughput
    --log-params-norm
    --timing-log-level 0
    --wandb-project vit-synth-rope-experiments
    --wandb-exp-name test
)


#rm -rf $CHECKPOINT_PATH/*
#rm -rf $TENSORBOARD_LOGS_PATH/*
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_vision.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
