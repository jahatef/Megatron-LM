#!/bin/bash

export GPUS_PER_NODE=2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NUM_NODES=$SLURM_JOB_NUM_NODES
export NODE_RANK=$SLURM_NODEID
export WORLD_SIZE=$(($GPUS_PER_NODE * $NUM_NODES))

IMG_SIZE=224
ROPE_TYPE="hilbert"
ROPE_BASE_TAG="16"

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
    --vit-rope-impl hilbert
    #--vit-rotary-base "10,20,30,40,50,60,70,80,90,100,110,120" \
    --vit-rotary-base 16
    --img-size $IMG_SIZE \
    --patch-dim 16 \
    --attention-backend flash # Can use (flash/fused/unfused/local)
    --transformer-impl transformer_engine
)

TRAINING_ARGS=(
    --micro-batch-size 128
    --global-batch-size 512
    --train-iters 553    
    --finetune
    --weight-decay 0.2 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 1.0e-4 
    --lr-decay-style cosine 
    --min-lr 1.0e-6
    --lr-warmup-fraction .005 
    --lr-decay-iters 553
    --recompute-granularity selective 
    #--profile
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
    --data-path "/home/hatef.4/datasets/inat-500"
    --dataloader-type 'cyclic'
    --vocab-file "/home/hatef.4/neox/gpt-neox/data/gpt2-vocab.json"
    --merge-file "/home/hatef.4/neox/gpt-neox/data/gpt2-merges.txt"
    --split 80,20,0
    --num-classes 500
)

EVAL_AND_LOGGING_ARGS=( 
    --log-interval 20
    --save-interval 79 
    --eval-interval 79 
    --eval-iters 10
    --tensorboard-log-interval 20
    --pretrained-checkpoint "/home/hatef.4/megatron/Megatron-LM/checkpoints-warmup-500"
    --tensorboard-dir "/home/hatef.4/megatron/Megatron-LM/logs"
    --ckpt-format torch
    --tensorboard-dir "/home/hatef.4/megatron/Megatron-LM/logs"
    --log-throughput
    --log-device-memory-used
    --log-num-zeros-in-grad
    --log-params-norm
    --timing-log-level 0
    --log-energy
    --wandb-project vit-synth-rope-experiments
    --save "/home/hatef.4/megatron/Megatron-LM/checkpoints-$ROPE_TYPE-$ROPE_BASE_TAG-$IMG_SIZE"
    --save-retain-interval 79
    --no-save-optim
    --no-save-rng
    --wandb-exp-name $ROPE_TYPE-$ROPE_BASE_TAG-$IMG_SIZE
)
export NCCL_DEBUG=WARN

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_vision.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} 
