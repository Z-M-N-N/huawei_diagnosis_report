#!/bin/bash
# ============================================================
# run_train.sh —— 超声报告生成模型 训练脚本（基于 VLM 框架）
# 项目: /home/mzhao/华为项目/Report
#
# 用法:
#   前台:  bash run_train.sh
#   后台:  cd /home/mzhao/华为项目/Report && \
#          setsid nohup bash run_train.sh >/dev/null 2>&1 &
# 说明: GPU_ID 由环境变量传入或在此修改（AGENTS 规则：用哪张卡必须用户指定）
#   GPU_ID=5 bash run_train.sh
# ============================================================
set -u

# ---------- conda 环境 ----------
source /opt/anaconda/etc/profile.d/conda.sh
conda activate Qwen_back

PROJ="/home/mzhao/华为项目/Report"
cd "$PROJ" || exit 1

# ---------- 配置区 ----------
train_mode="report"

export GPU_ID="${GPU_ID:-5}"        # ★ 训练用哪张卡（用户指定）
export GPU_NUM=1                    # 与 GPU_ID 个数一致

batch_size=1                        # 单卡 batch
grad_accum_steps=8                  # 梯度累积 → 有效 batch 8
lr=1e-4
epoch=1
save_step=500                       # 每 500 步保存 / 评估
seed=42

# 基座模型：第一阶段超声视觉适配后的权重（亦可换成原始 Qwen2.5-VL-7B）
llm="/home/mzhao/华为项目/VLM/Model/good_pvd_merged"

train_json="VLM/data/report/train.json"
lora_output_dir="VLM/lora/report"
deepspeed="VLM/scripts/zero3.json"

clean=0                             # 1=训练前清空 lora_output_dir；0=续训

# ---------- 派生 / 环境 ----------
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export NPROC_PER_NODE="$GPU_NUM"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}"
export PYTHONPATH="$PROJ:$PROJ/VLM"

# ---------- 日志（统一 /tmp/echo_logs/）----------
LOG_DIR="/tmp/echo_logs"
mkdir -p "$LOG_DIR"
log_file="${LOG_DIR}/logs_train_${train_mode}.log"

# ---------- 训练 ----------
if [ "$clean" = "1" ] && [ -d "$lora_output_dir" ]; then
    echo "🧹 clean=1，删除旧输出目录: $lora_output_dir"
    rm -rf "$lora_output_dir"
fi

echo "=========================================="
echo " 超声报告生成模型 训练配置"
echo "=========================================="
echo " 基座模型      : $llm"
echo " 训练数据      : $train_json"
echo " 验证数据      : ${train_json/train/test}"
echo " LoRA 输出     : $lora_output_dir"
echo " 有效 batch    : $((batch_size * grad_accum_steps))"
echo " lr / epoch    : $lr / $epoch   (save_step=$save_step, seed=$seed)"
echo " GPU           : $CUDA_VISIBLE_DEVICES (nproc=$NPROC_PER_NODE)"
echo " 日志          : $log_file"
echo "=========================================="
echo "开始时间: $(date)"
echo ""

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    VLM/train/train_qwen.py \
    --deepspeed $deepspeed \
    --model_name_or_path "$llm" \
    --dataset_use "ECHO_HEART_train%100" \
    --dataset_use_eval "ECHO_HEART_test%5" \
    --data_json "$train_json" \
    --data_flatten True \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --lora_enable True \
    --output_dir "$lora_output_dir" \
    --num_train_epochs $epoch \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --gradient_accumulation_steps $grad_accum_steps \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "steps" \
    --eval_steps $save_step \
    --save_strategy "steps" \
    --save_steps $save_step \
    --save_total_limit 20 \
    --learning_rate $lr \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 8 \
    --model_max_length 40960 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --seed $seed \
    --run_name "qwen2vl-${train_mode}" \
    --report_to none 2>&1 | tee "$log_file"

echo ""
echo "训练结束时间: $(date)"
echo "查看日志: tail -f $log_file"
