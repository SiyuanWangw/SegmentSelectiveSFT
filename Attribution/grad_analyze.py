import os
import json
import torch
from tqdm import tqdm
import argparse
import math
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class IntegratedGradientsAttribution:
    """
    Compute Integrated Gradients (IG) attributions from intermediate tokens
    to the model's final answer tokens.

    Notes:
      - We attribute the summed log-probability of answer tokens to input token embeddings.
      - The baseline is a sequence filled with `baseline_token_id` (often the pad token).
    """
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("pad_token_id", self.tokenizer.pad_token_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.model.eval()

    @torch.no_grad()
    def _embed(self, input_ids):
        return self.model.model.embed_tokens(input_ids)

    def compute_step_to_answer_attribution_integrated(self, input_ids, step_indices, answer_indices, baseline_token_id=0, steps=50):
        input_ids = torch.tensor(input_ids, dtype=torch.int32).unsqueeze(0).cuda()
        
        with torch.no_grad():
            input_embeddings = self._embed(input_ids)  # [1, L, D]
            baseline_embeddings = self._embed(torch.full_like(input_ids, baseline_token_id))      # [1, L, D]

        # Interpolation path: baseline -> input
        alphas = torch.linspace(0, 1, steps).view(-1, 1, 1, 1).to(torch.bfloat16).cuda()
        total_gradients = torch.zeros_like(input_embeddings) # [1, L, D]

        answer_token_ids = input_ids[0, answer_indices[0]:answer_indices[1]] # [A]
        
        for i in range(steps):
            interpolated_embedding = (baseline_embeddings + alphas[i] * (input_embeddings - baseline_embeddings)).detach()
            interpolated_embedding.requires_grad_(True)
        
            self.model.zero_grad()
            output = self.model(inputs_embeds=interpolated_embedding)
            logits = output.logits # [1, L, V]

            target_logits = logits[0, answer_indices[0]-1:answer_indices[1]-1] # [A, V]
            log_probs = torch.nn.functional.log_softmax(target_logits, dim=-1) # [A, V]
            target_log_probs = log_probs[range(len(answer_token_ids)), answer_token_ids]

            loss = target_log_probs.sum()
            loss.backward()

            total_gradients += interpolated_embedding.grad

        avg_gradients = total_gradients / steps
        input_diff = input_embeddings - baseline_embeddings  
        attributions = (input_diff * avg_gradients).sum(dim=-1).squeeze()
        attributions = attributions / attributions.norm()

        step_scores = []
        for step_idx in step_indices:
            step_score = attributions[step_idx[0]: step_idx[1]]
            step_scores.append(step_score.detach().cpu().float().tolist())

        del output, total_gradients, input_embeddings, baseline_embeddings
        torch.cuda.empty_cache()
        return step_scores
        

    def batch_compute_step_to_answer_attribution_integrated(
        self,
        input_ids,
        step_indices,
        answer_indices,
        baseline_token_id=0,
        steps=50,
        batch_size=1,   
    ):
        # [1, L]
        input_ids = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0).cuda()

        with torch.no_grad():
            input_embeddings = self._embed(input_ids)  # [1, L, D]
            baseline_embeddings = self._embed(torch.full_like(input_ids, baseline_token_id))      # [1, L, D]

        alphas = torch.linspace(0, 1, steps).view(steps, 1, 1).to(torch.bfloat16).cuda()
        total_gradients = torch.zeros_like(input_embeddings)  # [1, L, D]

        ans_start, ans_end = answer_indices
        answer_token_ids = input_ids[0, ans_start:ans_end]          # [A]

        step_pos = 0
        while step_pos < steps:
            chunk_end = min(step_pos + batch_size, steps)
            chunk_size = chunk_end - step_pos
            alphas_chunk = alphas[step_pos:chunk_end]  # [chunk_size, 1, 1]

            # Expand embeddings to [chunk_size, L, D]
            input_expand = input_embeddings.expand(chunk_size, -1, -1)
            baseline_expand = baseline_embeddings.expand(chunk_size, -1, -1)

            # Interpolation path: baseline -> input
            interpolated_embeddings = baseline_expand + alphas_chunk * (input_expand - baseline_expand).detach()
            interpolated_embeddings = interpolated_embeddings.to(dtype=torch.bfloat16)
            interpolated_embeddings.requires_grad_(True)

            # forward：batch = chunk_size
            self.model.zero_grad(set_to_none=True)
            output = self.model(inputs_embeds=interpolated_embeddings)
            logits = output.logits  # [chunk_size, L, V]

            # Collect logits corresponding to answer positions
            # target_logits: [chunk_size, A, V]
            target_logits = logits[:, ans_start-1:ans_end-1, :]
            log_probs = torch.nn.functional.log_softmax(target_logits, dim=-1)  # [chunk_size, A, V]

            # answer_token_ids: [A] -> [chunk_size, A]
            answer_ids_expand = answer_token_ids.unsqueeze(0).expand(chunk_size, -1)  # [chunk_size, A]

            target_log_probs = log_probs.gather(
                dim=-1,
                index=answer_ids_expand.unsqueeze(-1)
            ).squeeze(-1) # [chunk_size, A]

            loss = target_log_probs.sum()
            grads_chunk = torch.autograd.grad(
                loss,
                interpolated_embeddings,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]   # [chunk_size, L, D]

            # Sum gradients over the chunk's interpolation points
            total_gradients += grads_chunk.sum(dim=0, keepdim=True)  # -> [1, L, D]
            step_pos = chunk_end

        avg_gradients = total_gradients / steps  # [1, L, D]
        input_diff = input_embeddings - baseline_embeddings  # [1, L, D]
        attributions = (input_diff * avg_gradients).sum(dim=-1).squeeze(0)  # [L]
        attributions = attributions / attributions.norm()

        step_scores = []
        for step_idx in step_indices:
            step_attr = attributions[step_idx[0]: step_idx[1]]
            step_scores.append(step_attr.detach().cpu().float().numpy().tolist())

        del input_embeddings, baseline_embeddings, total_gradients
        torch.cuda.empty_cache()

        return step_scores
    

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True, help="HuggingFace model name or local path")
    p.add_argument("--input_data", type=str, required=True, help="Path to input jsonl")
    p.add_argument("--output_data_file", type=str, required=True, help="Path to output jsonl (appended)")
    p.add_argument("--output_ig_file", type=str, required=True, help="Path to output IG jsonl")
    p.add_argument("--ig_steps", type=int, default=20, help="Number of IG steps")
    # p.add_argument("--baseline_token", type=str, default="pad", choices=["pad", "zero"], help="Baseline token choice")
    return p.parse_args()


if __name__ == "__main__":
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

    args = parse_args()
    attribution_calculator = IntegratedGradientsAttribution(args.model_name)

    input_data = []
    with open(args.input_data, "r") as f:
        for line in f:
            json_obj = json.loads(line.strip())  
            input_data.append(json_obj)

    os.makedirs(os.path.dirname(args.output_data_file), exist_ok=True)
    with open(args.output_data_file, 'a') as f:
        input_template = "{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        for n in tqdm(range(len(input_data))): 
            each_data = input_data[n]
            user_msg = input_template.format(input=each_data["question"])
            user_tokens = attribution_calculator.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=True,
                add_generation_prompt=True,
            )
            
            pred_thoughts = each_data["segments"]  
            assistant_token_spans = []
            cursor = 0
            for k in range(len(pred_thoughts)):
                start = cursor
                segments = "".join(pred_thoughts[:k+1])
                end = len(attribution_calculator.tokenizer(segments, add_special_tokens=False)["input_ids"])
                assistant_token_spans.append((start, end))
                cursor = end
            assistant_tokens = attribution_calculator.tokenizer("".join(pred_thoughts), add_special_tokens=False)["input_ids"] 

            # Shift segment spans by user prompt length
            offset = len(user_tokens)
            adjusted_spans = [(start + offset, end + offset) for (start, end) in assistant_token_spans]

            answer_string = "</think> So, the final answer is \\boxed{" + each_data['answer'] + "}"
            
            answer_tokens = attribution_calculator.tokenizer(answer_string, add_special_tokens=False)["input_ids"] 
            answer_tokens_split = attribution_calculator.tokenizer.convert_ids_to_tokens(answer_tokens)

            for token_index in range(len(answer_tokens_split)):
                if 'boxed' in answer_tokens_split[token_index]:
                    ans_start = token_index + 1
            assert '{' in answer_tokens_split[ans_start]
            ans_end = len(answer_tokens_split)
            is_end = False
            stack = 0
            for m in range(ans_start, len(answer_tokens_split)):
                if '{' in answer_tokens_split[m] or '}' in answer_tokens_split[m]:
                    for each_tok in answer_tokens_split[m]:
                        if each_tok == "{":
                            stack += 1
                        elif each_tok == "}":
                            stack -= 1
                            if stack == 0:
                                ans_end = m #+1
                                is_end = True
                                break
                    if is_end:
                        break
            
            full_tokens = user_tokens + assistant_tokens + answer_tokens
            answer_indices = (ans_start + 1 + len(user_tokens + assistant_tokens), ans_end + len(user_tokens + assistant_tokens))

            print("answer tokens", answer_tokens_split[ans_start: ans_end+1], adjusted_spans)
            # importance_scores = attribution_calculator.compute_step_to_answer_attribution_integrated(full_tokens, adjusted_spans, answer_indices, baseline_token_id=attribution_calculator.tokenizer.pad_token_id, steps=args.ig_steps)
            importance_scores = attribution_calculator.batch_compute_step_to_answer_attribution_integrated(full_tokens, adjusted_spans, answer_indices, baseline_token_id=attribution_calculator.tokenizer.pad_token_id, steps=args.ig_steps)   
            input_data[n]["attribution"] = importance_scores

            f.write(json.dumps(input_data[n], ensure_ascii=False) + '\n') 

   
    attribution_output_data = []     
    with open(args.output_data_file, 'r') as f: 
        for line in f:
            json_obj = json.loads(line.strip())  
            attribution_output_data.append(json_obj)
    print(len(attribution_output_data)) 

    all_IG = []
    for n, each_data in enumerate(attribution_output_data):
        all_IG.append(each_data["attribution"])
    with open(args.output_ig_file, "w") as f:
        for each in all_IG:
            f.write(json.dumps(each, ensure_ascii=False) + '\n') 
    