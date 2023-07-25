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
DATASET_NAME="yentinglin/zh_TW_c4"

export WANDB_PROJECT="Chinese-LLAMA2"
export WANDB_TAGS="${MODEL_NAME},${DATASET_NAME},${MODEL_SIZE},fsdp,longchat"

echo "Training llama2 model: ${MODEL_NAME}"
echo "using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"


python -m torch.distributed.run --nproc_per_node=8 \
        longchat/train/pretrain/pretrain_flash.py \
        --model_name_or_path $MODEL_NAME \
        --data_path data/dummy_conversation.json  \
        --bf16 \
        --output_dir outputs \
        --num_train_epochs 1    \
        --per_device_train_batch_size $BATCH_SIZE_PER_GPU  \
        --per_device_eval_batch_size $BATCH_SIZE_PER_GPU  \
        --gradient_accumulation_steps $GRADIENT_ACC_STEPS  \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 10000  \
        --save_total_limit 2 \
        --learning_rate 2e-5 \
        --weight_decay 0.  \
        --warmup_ratio 0.03  \
        --lr_scheduler_type "cosine" \
        --logging_steps 1  \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True  \
        --model_max_length 2048  \
        --gradient_checkpointing True  \
        --lazy_preprocess True
