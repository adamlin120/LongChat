export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# first cli argument is the model size, default is 7
# second cli argument is the batch size per gpu, default is 1
MODEL_SIZE_ARG=${1:-7}
BATCH_SIZE_PER_GPU=${2:-1}
DEBUG=${3:-0}
DEEPSPEED=${4:-0}

MODEL_SIZE="${MODEL_SIZE_ARG}b"
NUM_GPUS=8
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

MODEL_NAME="meta-llama/Llama-2-${MODEL_SIZE}-hf"

export WANDB_PROJECT="Chinese-LLAMA2"
export WANDB_TAGS="${MODEL_NAME},zh_c4,zh_wiki,en_wiki,${MODEL_SIZE},fsdp,trl"

echo "Training llama2 model: ${MODEL_NAME}"
echo "using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"


# if deepspeed is 1, use deepspeed to train, else use torch.distributed
# if deepspeed is 1, use deepspeed to train, else use torch.distributed

if [ "$DEEPSPEED" -eq 1 ]; then
    accelerate launch \
      --mixed_precision bf16 \
      --num_machines 1 \
      --num_processes $NUM_GPUS \
      --use_deepspeed \
      --deepspeed_config_file stage3_no_offloading.conf \
      sft_trainer.py \
        --model_name $MODEL_NAME \
        --output_dir "zh_llama2_${MODEL_SIZE}" \
        --debug $DEBUG \
        --learning_rate 1e-5
else
    python -m torch.distributed.run \
      --nproc_per_node=$NUM_GPUS \
      sft_trainer.py \
      --model_name $MODEL_NAME \
      --output_dir "zh_llama2_${MODEL_SIZE}" \
      --debug $DEBUG \
      --learning_rate 1e-5
fi

#python -m torch.distributed.run \
#  --nproc_per_node=8 \
#  sft_trainer.py \
#  --model_name $MODEL_NAME \
#  --output_dir "zh_llama2_${MODEL_SIZE}" \
#  --debug $DEBUG \
#  --learning_rate 1e-5
#
