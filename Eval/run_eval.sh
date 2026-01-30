#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"
export TOKENIZERS_PARALLELISM=false

task_configurations=(
    "aime24 32" 
    "amc23 32" 
    "math500 6" 
    "minerva 6" 
    "gpqa 6" 
    "olympiad 6" 
)

# =========================
# Configuration
# =========================
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
OUTPUT_ROOT="outputs_origin"
SEED=0
MAX_TOKENS=32768
TEMPERATURE=0.6

for config in "${task_configurations[@]}"; do
  read -r task sample_n <<< "$config"

  echo "=============================================="
  echo "Running task: ${task}"
  echo "Samples per query: ${sample_n}"
  echo "=============================================="

  python -u math_eval.py \
      --model_name_or_path "${MODEL_PATH}" \
      --data_name "${task}" \
      --output_dir "${OUTPUT_ROOT}/${task}/r1_qwen_7b" \
      --split "test" \
      --prompt_type "deepseek-longcot" \
      --num_test_sample -1 \
      --max_tokens_per_call "${MAX_TOKENS}" \
      --seed "${SEED}" \
      --temperature "${TEMPERATURE}" \
      --n_sampling "${sample_n}" \
      --top_p 1 \
      --start 0 \
      --end -1 \
      --use_vllm \
      --save_outputs \
      --apply_chat_template
done
