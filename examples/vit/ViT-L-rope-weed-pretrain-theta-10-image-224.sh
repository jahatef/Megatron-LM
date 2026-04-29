#!/bin/bash

export GPUS_PER_NODE=2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NUM_NODES=$SLURM_JOB_NUM_NODES
export NODE_RANK=$SLURM_NODEID
export WORLD_SIZE=$(($GPUS_PER_NODE * $NUM_NODES))

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
    --vit-rope-impl axial
    #--vit-rotary-base "10,20,30,40,50,60,70,80,90,100,110,120" \
    --vit-rotary-base 10
    --img-size 224 \
    --patch-dim 16 \
    --attention-backend flash # Can use (flash/fused/unfused/local)
    --transformer-impl transformer_engine
)

TRAINING_ARGS=(
    --micro-batch-size 128 
    --global-batch-size 256
    --train-iters 480 
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
    --lr-decay-iters 480 
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
    --data-path "/home/hatef.4/datasets/harvest-datasets/weed/combined"
    --dataloader-type 'cyclic'
    --vocab-file "/home/hatef.4/neox/gpt-neox/data/gpt2-vocab.json"
    --merge-file "/home/hatef.4/neox/gpt-neox/data/gpt2-merges.txt"
    --split 80,20,0
    --num-classes 4
)

EVAL_AND_LOGGING_ARGS=( 
    --log-interval 20
    --save-interval 48 
    --eval-interval 48 
    --eval-iters 6
    --tensorboard-log-interval 20
    --save "/home/hatef.4/megatron/Megatron-LM/checkpoints-weed-warmup-224-test"
    --pretrained-checkpoint "/home/hatef.4/megatron/Megatron-LM/checkpoints-weed"
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
    --wandb-exp-name warmup-weed-224-test
)
export NCCL_DEBUG=WARN

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_vision.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} 
