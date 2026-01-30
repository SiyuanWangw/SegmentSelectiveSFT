# Segment-Level Attribution for Selective Learning of Long Reasoning Traces

**Authors:** Siyuan Wang, Yanchen Liu, Xiang Ren

> This repository contains the implementation for our ICLR 2026 paper that introduces a segment-level attribution framework for selectively learning from long chain-of-thought (CoT) reasoning traces. The pipeline consists of three major stages:  
> 1. IG Attribution Calculation and Important Segments Identification (`Attribution`)
> 2. Selective Supervised Fine-Tuning (`SelectiveSFT`)  
> 3. Evaluation across Benchmarks (`Eval`)


## Environment Setup
The code is tested with Python 3.11. We recommend using Conda for environment management.

```bash
# 1. Create a conda environment
conda create -n selective_sft python=3.11
conda activate selective_sft

# 2. Install standard dependencies
pip install -r requirements.txt

# 3 Install latex2sympy
cd AttributionSelectiveSFT/Eval/latex2sympy
pip install -e .

# 4. Install dependencies for unsloth 
cd SelectiveSFT
pip install -r requirements.txt
```

## Data Preparation
We utilize long CoT traces from the LIMO dataset for training. Then we evaluate on both in-domain benchmarks (MATH500, AMC23, AIME24) and out-of-domain benchmarks (GPQA-Diamond, Minerva, OlympiadBench). The data are stored under the `data` folder.
```
data/
├── aime24
├── amc23
├── gpqa
├── limo
├── math500
├── minerva
├── olympiad
```

You can also self-generate CoT traces for the LIMO dataset and utilize the shortest one:
```bash
cd Eval
bash CoT_generation.sh
```

## Attribution
The attribution stage identifies important reasoning segments in long CoT traces using integrated gradient-based analysis.
```
Attribution/
├── processed_data/
├── cal_attribution.sh
├── get_important_segments.py
├── grad_analyze.py
├── segment_split.py
```

You can directly run `cal_attribution.sh`:
```bash
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
```

## SelectiveSFT
This stage performs selective supervised fine-tuning using only the identified important reasoning segments.
```
SelectiveSFT/
├── checkpoints/
├── run_train.sh
├── train_mask.py
├── requirements.txt
```

**Run Training**
```bash
cd SelectiveSFT
bash run_train.sh
```

## Eval
The module supports evaluation across datasets by running `bash run_eval.sh`.


## Citation

If you find this work useful, please cite:
```bibtex
@inproceedings{wang2026segment,
  title     = {Segment-Level Attribution for Selective Learning of Long Reasoning Traces},
  author    = {Wang, Siyuan and Liu, Yanchen and Ren, Xiang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```