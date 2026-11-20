#!/bin/bash
# ============================================================
# run_eval_report.sh —— 对指定 LoRA checkpoint 做固定子集评测并生成报告
# 用法: EVAL_GPU=5 bash run_eval_report.sh <TAG> <LORA_PATH>
#   TAG       : 输出子目录/标签，如 ckpt-1500
#   LORA_PATH : LoRA 目录；留空则评测基座模型
# ============================================================
set -uo pipefail
source /opt/anaconda/etc/profile.d/conda.sh
conda activate Qwen_back

PROJ="/home/mzhao/华为项目/Report"
cd "$PROJ" || exit 1

TAG="${1:?需要 TAG}"
LORA_PATH="${2:-}"
EVAL_GPU="${EVAL_GPU:-5}"
export CUDA_VISIBLE_DEVICES="$EVAL_GPU"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL_PATH="/home/mzhao/华为项目/VLM/Model/good_pvd_merged"
SUBSET="VLM/data/report/eval_subset.json"
OUT_DIR="out/report/${TAG}"
NUM_FRAMES="${NUM_FRAMES:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"

LOG_DIR="/tmp/echo_logs"; mkdir -p "$LOG_DIR"
log_file="${LOG_DIR}/logs_eval_${TAG}.log"

echo "==== 评测 $TAG (LoRA=${LORA_PATH:-<base>}, GPU=$EVAL_GPU) ====" | tee "$log_file"
/home/mzhao/.conda/envs/Qwen_back/bin/python -u VLM/src/inference_report.py \
    --model_path "$MODEL_PATH" \
    --lora_path "$LORA_PATH" \
    --subset_json "$SUBSET" \
    --save_results "$OUT_DIR" \
    --num_frames "$NUM_FRAMES" \
    --max_new_tokens "$MAX_NEW_TOKENS" 2>&1 | tee -a "$log_file"
if [ ! -s "$OUT_DIR/preds.json" ]; then
    echo "[ERROR] 推理失败：未生成 $OUT_DIR/preds.json" | tee -a "$log_file"; exit 1
fi

/home/mzhao/.conda/envs/Qwen_back/bin/python -u VLM/src/eval_report_metrics.py \
    --preds "$OUT_DIR/preds.json" \
    --out_dir "$OUT_DIR" 2>&1 | tee -a "$log_file"

/home/mzhao/.conda/envs/Qwen_back/bin/python -u VLM/src/make_report_md.py \
    --preds "$OUT_DIR/preds.json" \
    --metrics "$OUT_DIR/metrics.json" \
    --out_md "$OUT_DIR/report.md" \
    --tag "$TAG" --checkpoint "$LORA_PATH" --num_frames "$NUM_FRAMES" 2>&1 | tee -a "$log_file"

/home/mzhao/.conda/envs/Qwen_back/bin/python -u VLM/src/make_report_xlsx.py \
    --preds "$OUT_DIR/preds.json" \
    --metrics "$OUT_DIR/metrics.json" \
    --out_xlsx "$OUT_DIR/report.xlsx" \
    --tag "$TAG" --checkpoint "$LORA_PATH" --num_frames "$NUM_FRAMES" 2>&1 | tee -a "$log_file"

echo "报告: $OUT_DIR/report.md ; $OUT_DIR/report.xlsx" | tee -a "$log_file"
