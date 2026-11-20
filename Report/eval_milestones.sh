#!/bin/bash
# ============================================================
# eval_milestones.sh —— 训练里程碑自动评测（1/3、1/2、结束）
# 在后台运行；检测到目标 checkpoint 后自动评测并出报告。
# 用法: EVAL_GPU=5 setsid nohup bash eval_milestones.sh >/dev/null 2>&1 &
# ============================================================
set -u
PROJ="/home/mzhao/华为项目/Report"
cd "$PROJ" || exit 1
EVAL_GPU="${EVAL_GPU:-5}"
LOG=/tmp/echo_logs/logs_eval_milestones.log
mkdir -p /tmp/echo_logs
echo "[$(date)] 里程碑评测监控启动 (EVAL_GPU=$EVAL_GPU)" >> "$LOG"

LORA_ROOT="VLM/lora/report"

wait_ckpt_complete () {  # $1 = checkpoint dir
    local d="$1"
    while true; do
        if [ -f "$d/adapter_model.safetensors" ]; then
            local s1 s2
            s1=$(stat -c %s "$d/adapter_model.safetensors" 2>/dev/null || echo 0)
            sleep 20
            s2=$(stat -c %s "$d/adapter_model.safetensors" 2>/dev/null || echo 0)
            [ "$s1" = "$s2" ] && [ "$s1" != "0" ] && return 0
        fi
        sleep 30
    done
}

run_eval () {  # $1 tag  $2 lora_path  (失败自动重试最多3次)
    local tag="$1" lora="$2" ok=0 k
    for k in 1 2 3; do
        echo "[$(date)] 开始评测: $tag (第 $k 次, $lora)" >> "$LOG"
        if EVAL_GPU="$EVAL_GPU" bash run_eval_report.sh "$tag" "$lora" >> "$LOG" 2>&1 \
           && [ -s "out/report/$tag/preds.json" ] && [ -s "out/report/$tag/metrics.json" ]; then
            ok=1; echo "[$(date)] 完成评测: $tag" >> "$LOG"; break
        fi
        echo "[$(date)] 评测失败，等待重试: $tag" >> "$LOG"; sleep 60
    done
    [ "$ok" = "1" ] || echo "[$(date)] 评测最终失败: $tag" >> "$LOG"
    return 0
}

# ---- 里程碑1: 1/3 (约 step 1549) -> checkpoint-1500 ----
wait_ckpt_complete "$LORA_ROOT/checkpoint-1500"
[ -s "out/report/ckpt-1500/metrics.json" ] || run_eval "ckpt-1500" "$LORA_ROOT/checkpoint-1500"

# ---- 里程碑2: 1/2 (约 step 2324) -> checkpoint-2500 ----
wait_ckpt_complete "$LORA_ROOT/checkpoint-2500"
[ -s "out/report/ckpt-2500/metrics.json" ] || run_eval "ckpt-2500" "$LORA_ROOT/checkpoint-2500"

# ---- 里程碑3: 训练结束 -> 最终 LoRA (输出根目录) ----
while [ -f "$LORA_ROOT/adapter_model.safetensors" ] || pgrep -f train_qwen.py >/dev/null; do
    if [ -f "$LORA_ROOT/adapter_model.safetensors" ] && ! pgrep -f train_qwen.py >/dev/null; then break; fi
    sleep 60
done
# 等最终保存稳定
sleep 20
[ -s "out/report/final/metrics.json" ] || run_eval "final" "$LORA_ROOT"

echo "[$(date)] 三个里程碑评测全部完成" >> "$LOG"
