#!/bin/bash
# ============================================================
# run_eval.sh —— 超声报告生成模型 推理/评估脚本
# 用法: GPU_ID=5 bash run_eval.sh
# ============================================================
set -u
source /opt/anaconda/etc/profile.d/conda.sh
conda activate Qwen_back

PROJ="/home/mzhao/华为项目/Report"
cd "$PROJ" || exit 1

export GPU_ID="${GPU_ID:-5}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL_PATH="/home/mzhao/华为项目/VLM/Model/good_pvd_merged"
TEST_JSON="VLM/data/report/test.json"
LORA_PATH="VLM/lora/report"          # 改成具体 checkpoint-XXXX
SAVE_RESULTS="out/report"
SAMPLE_SIZE=999999
NUM_FRAMES=16

LOG_DIR="/tmp/echo_logs"; mkdir -p "$LOG_DIR"
log_file="${LOG_DIR}/logs_eval_report.log"

python -u VLM/src/inference_report.py \
    --model_path "$MODEL_PATH" \
    --test_json "$TEST_JSON" \
    --lora_path "$LORA_PATH" \
    --save_results "$SAVE_RESULTS" \
    --num_frames "$NUM_FRAMES" \
    --sample_size "$SAMPLE_SIZE" \
    2>&1 | tee "$log_file"

echo "预测: $SAVE_RESULTS/preds.json"
echo "日志: tail -f $log_file"
