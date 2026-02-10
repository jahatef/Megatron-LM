sudo docker run \
  --gpus=all \
  --rm -it \
  --ipc=host \
  --entrypoint bash \
  --workdir /workspace/megatron-lm \
  -v /home/jahatef/Documents/megatron-high-res/harvest-datasets/weed:/workspace/dataset \
  -v /home/jahatef/Documents/megatron-high-res/Megatron-LM:/workspace/megatron-lm \
  -v /home/jahatef/Documents/megatron-high-res/checkpoints:/workspace/checkpoints \
  megatron:latest

PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.01-py3
CHECKPOINT_PATH="/workspace/checkpoints" 
TENSORBOARD_LOGS_PATH="/workspace/megatron-lm/logs" #<Specify path>
VOCAB_FILE="/workspace/data/gpt2-vocab.json"
MERGE_FILE="/workspace/data/gpt2-merges.txt"
DATA_PATH="/workspace/dataset/"

bash examples/vit/train_gpt3_175b_distributed.sh $CHECKPOINT_PATH $TENSORBOARD_LOGS_PATH $VOCAB_FILE $MERGE_FILE $DATA_PATH 
