#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"
export HF_HUB_OFFLINE=1

# =========================
# Configuration
# =========================
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_PATH="../data/limo/solutions_top70cohe80_lennorm_7B_J50.jsonl"
EPOCHS=10
LR=3e-5

python -u train_mask.py \
    --model_name_or_path "${MODEL_PATH}" \
    --data_names "${DATA_PATH}" \
    --epochs "${EPOCHS}" \
    --learning_rate "${LR}" \
    --deepseek \
    --mask \
    --apply_all