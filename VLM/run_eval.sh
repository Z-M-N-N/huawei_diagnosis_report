#!/bin/bash

# ========== conda 环境 ==========
CONDA_BASE="/opt/anaconda"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate Qwen_back
else
    echo "警告: 未找到 ${CONDA_BASE}/etc/profile.d/conda.sh，跳过 conda 激活"
fi
# ========== 路径配置 ==========
EchoView_dir="/home/mzhao/ECHO_VIEW/VLM"
cd "${EchoView_dir}" || exit 1


export GPU_ID="4"                            # GPU ID 列表
export GPU_NUM="1"        
export CUDA_VISIBLE_DEVICES=$GPU_ID
export NPROC_PER_NODE=$GPU_NUM
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"



MODEL_PATH="Model/good_pvd_merged"
TEST_JSON="Data/pvd_no_mv/test.json"
SAVE_RESULTS="out/pvd_no_mv"

# LORA_PVD="lora/pvd/checkpoint-5000"
LORA_PVD=""
LORA_AVD="lora/avd"
LORA_CMD="lora/cmd"

# MODE="pvd"
MODE=""
DIAG_ITEM="all"
SAMPLE_SIZE=1000
NUM_FRAMES=16
TORCH_DTYPE="bfloat16"

# ========== 运行 ==========
python src/inference.py \
    --model_path "${MODEL_PATH}" \
    --test_json "${TEST_JSON}" \
    --diagnostic_mode "${MODE}" \
    --lora_path_PVD "${LORA_PVD}" \
    --lora_path_AVD "${LORA_AVD}" \
    --lora_path_CMD "${LORA_CMD}" \
    --diag_item "${DIAG_ITEM}" \
    --sample_size "${SAMPLE_SIZE}" \
    --num_frames "${NUM_FRAMES}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --save_results "${SAVE_RESULTS}"

python src/置信区间计算.py \
    --json_floder "${SAVE_RESULTS}" \
    --output "${SAVE_RESULTS}_CI.csv" 