import json
import numpy as np
import os
from transformers import AutoTokenizer
import math
import re
from tqdm import tqdm
import copy
tokenizer = AutoTokenizer.from_pretrained('../../models/DeepSeek-R1-Distill-Qwen-7B', trust_remote_code=True)

# segment keywords in our implementation
pattern = r"(\n\nWait|\n\nAlternatively|\n\nBut wait|\n\nBut alternatively|\n\nBut just to|\n\nHowever|\n\nNot sure|\n\nGoing back|\n\nBacktrack|\n\nTrace back|\n\nAnother)" #|\n\n\*\*Final Answer
# segment keywords in Retro-Search
# pattern = r"(\n\nWait|\n\nAlternatively|\n\nBut|\n\nHowever|\n\nHmmm|\n\nHmm|\n\nNot sure|\n\nGoing back|\n\nBacktrack|\n\nTrace back|\n\nAnother)" 
       
### Code for splitting solutions into segments    
input_data = []
with open('../data/limo/test.jsonl', 'r') as f:
    for line in f:
        json_obj = json.loads(line.strip())  
        input_data.append(json_obj)

segment_num = []
all_len = []
different_num = 0
for i, each_data in tqdm(enumerate(input_data)): 
    # cur_response = each_data['solution'].split("\n</think>")[0]
    # assert len(each_data['solution'].split("\n</think>")) == 1

    cur_response = each_data['solution']    
    # if len(each_data['solution'].split("\n\n**Final Answer**")) == 1:
    #     cur_response = each_data['solution']
    # else:
    #     assert len(each_data['solution'].split("\n\n**Final Answer**")) > 1
    #     cur_response = "\n\n**Final Answer**".join(each_data['solution'].split("\n\n**Final Answer**")[:-1])
    all_len.append(len(tokenizer(cur_response, add_special_tokens=False)['input_ids']))
            
    parts = re.split(pattern, cur_response)
    segments = [parts[0]]  
    for j in range(1, len(parts), 2):
        segments.append(parts[j] + parts[j + 1])  
    merged_segments = segments

    segment_num.append(len(merged_segments))  

    input_data[i]['segments'] = merged_segments

print(different_num, len(input_data))
print(sum(segment_num)/len(segment_num), np.max(all_len))

with open('../data/limo/solution_segments.jsonl', 'w') as f:
    for item in input_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n') 
