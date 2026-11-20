#!/bin/bash
# 合并多份 preds.json 并出报告(md+xlsx)
# 用法: bash merge_and_report.sh <TAG> <LORA_PATH> <cfg1.json> <cfg2.json> ...
set -uo pipefail
TAG="$1"; LORA="$2"; shift 2
OUT="out/report/${TAG}"; mkdir -p "$OUT"
PY=/home/mzhao/.conda/envs/Qwen_back/bin/python
$PY -u VLM/src/merge_preds.py --inputs "$@" --out "$OUT/preds.json"
$PY -u VLM/src/eval_report_metrics.py --preds "$OUT/preds.json" --out_dir "$OUT"
$PY -u VLM/src/make_report_md.py --preds "$OUT/preds.json" --metrics "$OUT/metrics.json" \
    --out_md "$OUT/report.md" --tag "$TAG" --checkpoint "$LORA" --num_frames 16
$PY -u VLM/src/make_report_xlsx.py --preds "$OUT/preds.json" --metrics "$OUT/metrics.json" \
    --out_xlsx "$OUT/report.xlsx" --tag "$TAG" --checkpoint "$LORA" --num_frames 16
echo "合并报告: $OUT/report.md ; $OUT/report.xlsx"
