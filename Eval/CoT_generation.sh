#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export TOKENIZERS_PARALLELISM=false

# =========================
# Configuration
# =========================
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DATA_NAME="limo"
OUTPUT_DIR="outputs/limo/r1_qwen_7b"
SEED=0
TEMPERATURE=0.6
N_SAMPLING=32
MAX_TOKENS=32768

python -u math_eval.py \
    --model_name_or_path "${MODEL_NAME}" \
    --data_name "${DATA_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --split "test" \
    --prompt_type "deepseek-longcot" \
    --num_test_sample -1 \
    --max_tokens_per_call "${MAX_TOKENS}" \
    --seed "${SEED}" \
    --temperature "${TEMPERATURE}" \
    --n_sampling "${N_SAMPLING}" \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --apply_chat_template
