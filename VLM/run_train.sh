#!/bin/bash
# ============================================================
# train.sh —— EchoView Qwen2.5-VL LoRA 训练脚本（无 tmux 版）
# 放在 EchoView-main/VLM_TRAIN 下运行
#
# 用法:
#   bash train.sh                                    # 前台跑
#   setsid nohup bash train.sh > run_all.log 2>&1 &  # 后台跑
# ============================================================

# ============ 1. conda / 路径探测 ============
if [ -f "/opt/anaconda/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="/opt/anaconda"
    EchoView_dir="/home/mzhao/ECHO_VIEW/VLM"
    Data_Path="../../Data/dicom_videos_group"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate Qwen_back
export CONDA_BASE

cd "${EchoView_dir}" || exit 1

# ============ 2. 配置区（统一在此修改） ============
export train_mode="pvd"                    # 数据/输出子目录名

export GPU_ID="2"                          # GPU ID 列表，如 "0,1,2,3"
export GPU_NUM="1"                         # GPU 数量，必须与 GPU_ID 个数一致
export batch_size=4                        # 单卡 batch size
export grad_accum_steps=1                  # 梯度累积步数
export lr=1e-4                             # 学习率
export epoch=1                             # 训练轮数
export save_step=1000                      # 每多少步保存/评估

export llm="Model/qwen2.5-vl-7b-visual_finetuned_cls_reg"   # 基座模型权重

export train_json="Data/pvd/train.json"
export lora_output_dir="lora/${train_mode}_test"

export CUDA_VISIBLE_DEVICES=$GPU_ID
export NPROC_PER_NODE=$GPU_NUM
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# DeepSpeed 配置
deepspeed="scripts/zero3.json"

# ============ 3. 日志配置 ============
log_dir="${lora_output_dir}/logs"
mkdir -p "$log_dir"

log_file="${log_dir}/train_${train_mode}.log"

# ============ 4. 训练 ============
run_train() {
    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    export MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

    echo "=========================================="
    echo "训练配置"
    echo "=========================================="
    echo " train_mode      : $train_mode"
    echo " llm             : $llm"
    echo " train_json      : $train_json"
    echo " lora_output_dir : $lora_output_dir"
    echo " lr              : $lr"
    echo " batch_size      : $batch_size"
    echo " grad_accum      : $grad_accum_steps"
    echo " effective_batch : $((batch_size * grad_accum_steps))"
    echo " epoch           : $epoch"
    echo " save_step       : $save_step"
    echo " GPU_ID          : $CUDA_VISIBLE_DEVICES"
    echo " NPROC_PER_NODE  : $NPROC_PER_NODE"
    echo " log_file        : $log_file"
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
        --dataset_use_eval "ECHO_HEART_test%100" \
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
        --save_total_limit 26 \
        --learning_rate $lr \
        --weight_decay 0.05 \
        --warmup_ratio 0.03 \
        --max_grad_norm 1 \
        --lr_scheduler_type "cosine" \
        --logging_steps 8 \
        --model_max_length 40960 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --run_name "qwen2vl-${train_mode}" \
        --report_to none
        # --report_to none 2>&1 | tee "$log_file"

    echo ""
    echo "=========================================="
    echo "训练结束时间: $(date)"
    echo "=========================================="
}

echo "=========================================="
echo "🚀 启动训练（前台 / nohup 后台均可）"
echo "=========================================="
echo "日志文件: $log_file"
echo "=========================================="

run_train

echo ""
echo "=============================================="
echo "🏁 训练脚本执行完毕: $(date)"
echo "=============================================="