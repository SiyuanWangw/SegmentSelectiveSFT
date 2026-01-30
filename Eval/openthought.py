from datasets import load_dataset
from tqdm import tqdm
import json
import os
from parser import extract_answer

# # Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("open-thoughts/OpenThoughts-114k", "metadata")['train']
# print(len(ds))

# selected_data = []
# source_list = ['numina_math', 'code_contests', 'apps', 'taco', 'codeforces', 'camelai_biology', 'camelai_physics', 'camelai_chemistry', 'riddle_sense']
# i = 0
# for each in tqdm(ds):
#     if each['source'] == source_list[0]:
#         # source_list.append(each['source'])
#         if i % 5 == 0: # 10
#             selected_data.append(each)
#         i += 1

def is_number(s):
    try:
        float(s)   # 尝试转成浮点数
        return True
    except ValueError:
        return False
        
# print(len(selected_data))    
# with open('../Eval/data/openthought/raw_selected_5.jsonl', 'w') as f:
#     for item in selected_data:
#         f.write(json.dumps(item, ensure_ascii=False) + '\n') 

# all_data = []
# with open('../Eval/data/openthought/raw_selected_5.jsonl', 'r') as f:
#     for line in f:
#         json_obj = json.loads(line.strip())  
#         all_data.append(json_obj)

# add_answer_data = []
# for each in tqdm(all_data):
#     answer = extract_answer(each['ground_truth_solution'], "math")
#     if is_number(answer) and len(answer) < 10:
#         each["answer"] = answer
#         add_answer_data.append(each)
# print(len(add_answer_data))

# with open('../Eval/data/openthought/number_selected_5.jsonl', 'w') as f:
#     for item in add_answer_data:
#         f.write(json.dumps(item, ensure_ascii=False) + '\n') 
        
all_data = []
question_list = []
with open('../Eval/data/limo/test.jsonl', 'r') as f:
    for line in f:
        json_obj = json.loads(line.strip())  
        all_data.append(json_obj)
        question_list.append(json_obj["question"].lower())

        
candidate_data = []
with open('../Eval/data/openthought/number_selected_5.jsonl', 'r') as f:
    for line in f:
        json_obj = json.loads(line.strip())  
        if "boxed{" in json_obj["deepseek_reasoning"]:
            candidate_data.append(json_obj)
print(candidate_data[0].keys(), len(all_data), len(candidate_data))

for each in candidate_data:
    if each["problem"].lower() not in question_list:
        cur_inst = {}
        cur_inst['question'] = each["problem"]
        cur_inst['solution'] = each["deepseek_reasoning"]
        cur_inst['answer'] = each["answer"]
        all_data.append(cur_inst)
print(len(all_data))

with open('../Eval/data/openthought/combine_data_5.jsonl', 'w') as f:
    for item in all_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n') 