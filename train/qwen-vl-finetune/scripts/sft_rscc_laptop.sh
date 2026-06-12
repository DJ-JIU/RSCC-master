#!/bin/bash

# Distributed training configuration - 单卡训练
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=1  # 只有1块RTX 5060

# DeepSpeed configuration（单卡不用也可以，但保留兼容性）
deepspeed=./scripts/zero3.json

# Model configuration - 使用3B模型适配7.5GB显存
# 使用本地缓存路径以避免网络问题
llm=/home/long/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3

# Training hyperparameters - 小batch适配小显存
lr=2e-7
batch_size=1
grad_accum_steps=8
# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration
datasets="rscc_subset%100"
data_flatten=False

# Output configuration
run_name="rscc_qwen25vl3b_sft"
output_dir="/home/long/RSCC-master/output/finetuned_models/rscc_qwen25vl3b"

# PYTHON env
PYTHON=/home/long/anaconda3/envs/genai/bin/python

# Training arguments (no extra quotes around string values)
args=(
    --model_name_or_path "${llm}"
    --cache_dir /home/long/.cache/huggingface
    --dataset_use ${datasets}
    --data_flatten ${data_flatten}
    --tune_mm_vision False
    --tune_mm_mlp True
    --tune_mm_llm True
    --bf16
    --output_dir ${output_dir}
    --num_train_epochs 5
    --per_device_train_batch_size ${batch_size}
    --per_device_eval_batch_size $((batch_size*2))
    --gradient_accumulation_steps ${grad_accum_steps}
    --max_pixels $((224*224*3))
    --min_pixels $((224*224*3))
    --eval_strategy no
    --save_strategy steps
    --save_steps 100
    --save_total_limit 2
    --learning_rate ${lr}
    --mm_projector_lr 1e-5
    --vision_tower_lr 1e-6
    --weight_decay 0.01
    --warmup_ratio 0.03
    --max_grad_norm 1
    --lr_scheduler_type cosine
    --logging_steps 1
    --model_max_length 2048
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --run_name ${run_name}
    --report_to none
    --optim adamw_torch
)

# Create output directory
mkdir -p /home/long/RSCC-master/output/finetuned_models

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="/home/long/RSCC-master/logs/finetune_${TIMESTAMP}.log"

echo "Starting training..."
echo "Log file: ${LOG_FILE}"

# Launch training
nohup ${PYTHON} -m torch.distributed.run \
    --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
    ${entry_file} "${args[@]}" > ${LOG_FILE} 2>&1 &

echo "Training started. Logs are being written to: ${LOG_FILE}"
echo "You can track progress with: tail -f ${LOG_FILE}"

