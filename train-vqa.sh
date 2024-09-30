# 下载链接： https://hf-mirror.com/Qwen/Qwen2-VL-7B-Instruct
pretrained_model=Qwen2-VL-7B-Instruct

SIZE_FACTOR=8 MAX_PIXELS=602112 CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 swift sft \
  --model_type  qwen2-vl-7b-instruct \
  --model_id_or_path $pretrained_model \
  --output_dir output/qwen2-vl-7b \
  --sft_type lora \
  --dtype bf16 \
  --num_train_epochs 3 \
  --learning_rate 5e-05 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --lr_scheduler_type cosine \
  --max_grad_norm 10.0 \
  --lora_rank 8 \
  --logging_steps 50 \
  --save_steps 200 \
  --eval_steps 200 \
  --save_total_limit 10 \
  --dataset data/instruction_train.json \
  --val_dataset data/instruction_dev.json \
  --report_to tensorboard


# # 合并权重推理
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/qwen2-vl-7b/qwen2-vl-7b-instruct/v0-20240928-181138/checkpoint-200 \
    --dtype bf16 \
    --load_dataset_config true --merge_lora true

