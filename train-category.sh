####################################类别########################################
# 预训练模型下载链接：https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct
pretrained_model=Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 swift sft \
  --model_type  qwen2_5-7b-instruct \
  --model_id_or_path $pretrained_model \
  --output_dir output/Qwen2.5-7B-Instruct_category \
  --sft_type lora \
  --dtype bf16 \
  --max_length 512 \
  --num_train_epochs 5 \
  --learning_rate 5e-05 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --lr_scheduler_type cosine \
  --max_grad_norm 10.0 \
  --lora_rank 32 \
  --logging_steps 20 \
  --save_steps 100 \
  --eval_steps 100 \
  --save_total_limit 5 \
  --dataset data/instruction_category_train.json \
  --val_dataset data/instruction_category_dev.json \
  --report_to tensorboard


# 注意需要修改ckpt_dir的路径，路径中含有时间戳信息，需要替换复现训练的路径
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/Qwen2.5-7B-Instruct_category/qwen2_5-7b-instruct/v0-20240924-100533/checkpoint-500 \
    --dtype bf16 \
    --load_dataset_config true --merge_lora true

