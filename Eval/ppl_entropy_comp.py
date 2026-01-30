from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math
import json
from tqdm import tqdm
import torch.nn.functional as F
import pickle

model_name = '../../models/DeepSeek-R1-Distill-Qwen-7B' # 14B
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
model.gradient_checkpointing_enable()
model.eval()

loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

input_data = []
with open("./data/limo/test_shortest_iter1_7b_n32topp1_segments.jsonl", "r") as f: #test_shortest_segment_v2
    for line in f:
        json_obj = json.loads(line.strip())  
        input_data.append(json_obj)

input_template = "{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
text_list = []
token_spans_list = []
for each_data in input_data:
    user_msg = input_template.format(input=each_data["question"])
    user_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=True,
        add_generation_prompt=True,
    )

    pred_thoughts = each_data["pred_thought"]  
    assistant_token_spans = []
    cursor = 0
    for n, seg in enumerate(pred_thoughts):
        start = cursor
        segments = "".join(pred_thoughts[:n+1])
        end = len(tokenizer(segments, add_special_tokens=False)["input_ids"])
        assistant_token_spans.append((start, end))
        cursor = end

    full_tokens = user_tokens + tokenizer(each_data["shortest_solution"], add_special_tokens=False)["input_ids"]
    text_list.append(full_tokens)

    offset = len(user_tokens)
    adjusted_spans = [(start + offset, end + offset) for (start, end) in assistant_token_spans]
    token_spans_list.append(adjusted_spans)


# with open("./processed_data/limo/test_shortest_segment_v2_ppl_entropy.jsonl", 'w') as f:
all_list = []
for i, input_ids in tqdm(enumerate(text_list)):
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).cuda() 

    # 只计算从 start_token_position 开始部分的 loss
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=False, output_hidden_states=False)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    # 换算维度： [batch, seq_len] → [seq_len]
    losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # print(losses.size())
    # 只取你关心的区域
    ppl_list = []
    for each_start, each_end in token_spans_list[i]:
        relevant_losses = losses[each_start-1: each_end-1]
        ppl_list.append(relevant_losses.to(torch.float32).cpu().numpy())

    probs = F.softmax(shift_logits, dim=-1)  # (1, T-1, V)
    # 计算每个位置的 entropy
    log_probs = torch.log(probs + 1e-9)      # 避免 log(0)
    entropy = -torch.sum(probs * log_probs, dim=-1).squeeze()  # (1, T-1)
    # print(entropy, entropy.size())
    entropy_list = []
    for each_start, each_end in token_spans_list[i]:
        relevant_entropy = entropy[each_start-1: each_end-1]
        # print(relevant_entropy)
        entropy_list.append(relevant_entropy.to(torch.float32).cpu().numpy())

    all_list.append([ppl_list, entropy_list])
    del outputs, shift_logits, shift_labels, losses, probs
    torch.cuda.empty_cache()
    
# print(all_list[0][0]) 
# print(all_list[0][1]) 
print(len(all_list))
with open("./processed_data/limo_7b/test_shortest_iter1_7b_n32topp1_segments_ppl_entropy.pkl", 'wb') as f:
    pickle.dump(all_list, f)