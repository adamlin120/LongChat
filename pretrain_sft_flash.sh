export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# first cli argument is the model size, default is 7
# second cli argument is the batch size per gpu, default is 1
MODEL_SIZE_ARG=${1:-7}
BATCH_SIZE_PER_GPU=${2:-1}

MODEL_SIZE="${MODEL_SIZE_ARG}b"
NUM_GPUS=8
TOTAL_BATCH_SIZE=8
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

MODEL_NAME="meta-llama/Llama-2-${MODEL_SIZE}-hf"

export WANDB_PROJECT="Chinese-LLAMA2"
export WANDB_TAGS="${MODEL_NAME},zh_c4,zh_wiki,en_wiki,${MODEL_SIZE},fsdp,trl"

echo "Training llama2 model: ${MODEL_NAME}"
echo "using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"


python -m torch.distributed.run \
  --nproc_per_node=8 \
  sft_trainer.py \
  --model_name $MODEL_NAME \
  --output_dir "zh_llama2/${MODEL_SIZE}"

