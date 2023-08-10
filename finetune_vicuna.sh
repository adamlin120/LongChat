export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

export WANDB_PROJECT="Chinese-LLAMA2"
export WANDB_TAGS="news,sft"

export WANDB_API_KEY="94f8e06129c90551b50bfc5556e389fc574c2fa3"
export HUGGING_FACE_HUB_TOKEN="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"

python -m torch.distributed.run --nproc_per_node=6 \
         longchat/train/fine_tune/finetune.py \
        --model_name_or_path "yentinglin/210k" \
        --data_path "zh_tw_instruction_sharegpt_format.json"  \
        --bf16 \
        --output_dir "news-pretrained-sft" \
        --num_train_epochs 3    \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 4  \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 10000  \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0.  \
        --warmup_ratio 0.03  \
        --lr_scheduler_type "cosine" \
        --logging_steps 1  \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True  \
        --model_max_length 1024  \
        --gradient_checkpointing True  \
        --lazy_preprocess True \
        --report_to wandb
