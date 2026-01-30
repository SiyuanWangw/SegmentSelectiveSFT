#!/usr/bin/env bash

# split solutions into segments
python segment_split.py

# calculate token attribution
export CUDA_VISIBLE_DEVICES="0,1"

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" # 1.5B
INPUT_DATA="../data/limo/solution_segments.jsonl"
OUTPUT_DATA_FILE="./processed_data/limo/solution_segments_attn_integ50_7b.jsonl"
OUTPUT_IG_FILE="./processed_data/limo/IG_7B_J50.jsonl"
IG_STEPS=50  # 20

python grad_analyze.py \
  --model_name "${MODEL_NAME}" \
  --input_data "${INPUT_DATA}" \
  --output_data_file "${OUTPUT_DATA_FILE}" \
  --output_ig_file "${OUTPUT_IG_FILE}" \
  --ig_steps "${IG_STEPS}" 


# aggregate token attributions and identify important segments
TRAINING_FILE="../data/limo/solutions_top70cohe80_lennorm_7B_J50.jsonl"
python get_important_segments.py \
  --input_data_file "${INPUT_DATA}" \
  --IG_score_data_file "${OUTPUT_IG_FILE}" \
  --output_data_file "${TRAINING_FILE}" 