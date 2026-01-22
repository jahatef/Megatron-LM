#!/bin/bash

PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.01-py3
CHECKPOINT_PATH="/workspace/checkpoints" 
TENSORBOARD_LOGS_PATH="/workspace/megatron-lm/logs" #<Specify path>
VOCAB_FILE="/workspace/data/gpt2-vocab.json"
MERGE_FILE="/workspace/data/gpt2-merges.txt"
DATA_PATH="/workspace/dataset/processed_data_text_document" #<Specify path and file prefix>_text_document

sudo docker run \
  --gpus=all \
  --rm -it \
  --ipc=host \
  --entrypoint bash \
  --workdir /workspace/megatron-lm \
  -v /home/jahatef/Documents/megatron-high-res/data/enwik8:/workspace/dataset \
  -v /home/jahatef/Documents/megatron-high-res/Megatron-LM:/workspace/megatron-lm \
  -v /home/jahatef/Documents/megatron-high-res/checkpoints:/workspace/checkpoints \
  -v /home/jahatef/Documents/megatron-high-res/data:/workspace/data \
  megatron:latest \
  bash examples/gpt3/train_gpt3_175b_distributed.sh $CHECKPOINT_PATH $TENSORBOARD_LOGS_PATH $VOCAB_FILE $MERGE_FILE $DATA_PATH 

'''
python tools/preprocess_data.py \
    --input /workspace/data/enwik8/enwik8.jsonl \
    --output-prefix /workspace/data/processed_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model gpt2 \
    --workers 1 \
    --append-eod
'''