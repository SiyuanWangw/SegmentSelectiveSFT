import json
import math
import numpy as np
import random
import argparse
import re
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_data_file", type=str, required=True, help="Path to input jsonl")
    p.add_argument("--IG_score_data_file", type=str, required=True, help="Path to input IG scores")
    p.add_argument("--output_data_file", type=str, required=True, help="Path to output jsonl (appended)")
    return p.parse_args()

args = parse_args()

input_data = []
with open(args.input_data_file, "r") as f: 
    for line in f:
        json_obj = json.loads(line.strip())  
        input_data.append(json_obj)

all_IG_list = []
with open(args.IG_score_data_file, "r") as f: 
    for line in f:
        json_obj = json.loads(line.strip())  
        all_IG_list.append(json_obj)
print("sample number", len(input_data), len(all_IG_list))


ratio_list = []
for i in range(len(input_data)):
    assert len(all_IG_list[i]) == len(input_data[i]["segments"]), f"{i} {len(all_IG_list[i])}. {len(input_data[i]['segments'])}"
    cur_IGs = all_IG_list[i]
        
    all_segs_IG_stres = []
    for each_seg in cur_IGs:
        abs_each_seg = [abs(_) for _ in each_seg]
        all_segs_IG_stres.append(np.sum(abs_each_seg)/(len(abs_each_seg)**0.5))
    indexed_sorted = sorted(enumerate(all_segs_IG_stres), key=lambda x: -x[1])
    sorted_indices = [idx for idx, val in indexed_sorted]
    sorted_inst_IG_stre = np.array([val for idx, val in indexed_sorted])

    normed = sorted_inst_IG_stre / sorted_inst_IG_stre.sum()
    cumsum = np.cumsum(normed)
    for j in range(len(cumsum)):
        if cumsum[j] >= 0.7:
            break
    important_index = sorted(sorted_indices[:j+1])
        
    IG_dire_list = []
    for _ in range(len(input_data[i]["segments"])):
        IG_dire_list.append(abs(np.sum(cur_IGs[_])) / np.sum(np.abs(cur_IGs[_])))  
    
    select_span_ids = [_ for _ in important_index if IG_dire_list[_] <= 0.8]
    assert select_span_ids == sorted(select_span_ids)
    
    input_data[i]['selected_spans_ids'] = select_span_ids
    ratio_list.append(len(select_span_ids)/len(cur_IGs))
        
with open(args.output_data_file, 'w') as f:
    for n in range(len(input_data)):
        f.write(json.dumps(input_data[n], ensure_ascii=False) + '\n')
print(np.mean(ratio_list))



