import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
from unsloth import FastLanguageModel 
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import torch
import argparse
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from unsloth import is_bfloat16_supported
import re
import gc
pattern = r"(\n\nWait|\n\nAlternatively|\n\nBut wait|\n\nBut alternatively|\n\nBut just to|\n\nHowever|\n\nNot sure|\n\nGoing back|\n\nBacktrack|\n\nTrace back|\n\nAnother)" #|\n\n\*\*Final Answer
# pattern = r"(\n\nWait|\n\nAlternatively|\n\nBut|\n\nHowever|\n\nHmmm|\n\nHmm|\n\nNot sure|\n\nGoing back|\n\nBacktrack|\n\nTrace back|\n\nAnother)" 
os.environ["WANDB_PROJECT"] = "***" 


class EarlyStopAtEpochCallback(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.epoch >= 9:
            print(f"Epoch {state.epoch:.1f} reached. Stopping training early.")
            control.should_training_stop = True
        return control


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="GAIR/LIMO", type=str)
    parser.add_argument("--model_name_or_path", default="unsloth/Qwen2.5-7B-Instruct", type=str)
    parser.add_argument("--output_dir", default="./outputs_qwen25_7b_instruct_no4bit_epoch5_lr5e-6", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=16384) #16384 32768
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--deepseek", action="store_true")
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--apply_all", action="store_true")

    args = parser.parse_args()
    return args

args = parse_args()
print(args.load_in_4bit, args.deepseek)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model_name_or_path,
    max_seq_length = args.max_seq_length, # Choose any for long context!
    load_in_4bit = args.load_in_4bit,  # 4 bit quantization to reduce memory
    full_finetuning = True, # [NEW!] We have full finetuning now!
)

if not args.deepseek:
    args.run_name = args.model_name_or_path.split("/")[-1] + "max_seq_" + str(args.max_seq_length) + "lr_" + str(args.learning_rate) + "epochs_" + str(args.epochs)
    args.output_dir = f"./checkpoints/{args.model_name_or_path.split('/')[-1]}_epoch{args.epochs}_lr{args.learning_rate}"
    instruction_template = "<|im_start|>user"
    response_template = "<|im_start|>assistant\n"+"<think>\n"
    
    tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>", "<|reason_pad|>"]})
    model.resize_token_embeddings(len(tokenizer))
else:
    args.run_name = args.model_name_or_path.split("/")[-1] + "max_seq_" + str(args.max_seq_length) + "lr_" + str(args.learning_rate) + "epochs_" + str(args.epochs)
    args.output_dir = f"./checkpoints/{args.model_name_or_path.split('/')[-1]}_epoch{args.epochs}_lr{args.learning_rate}_len{args.max_seq_length}"
    instruction_template = "<｜begin▁of▁sentence｜><｜User｜>"
    response_template = "<｜Assistant｜><think>\n"

model.config.use_cache = False
model.gradient_checkpointing_enable()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

def formatting_prompts_func(examples):
    questions = examples["question"]
    outputs = examples["solution"] 
    segments_ids = examples["selected_spans_ids"]

    print(len(outputs), len(questions), len(segments_ids))

    input_ids_list = []
    labels_list = []
    num = 0
    for prompt, output, segment_id in zip(questions, outputs, segments_ids):
        messages = [
            {"role": "user", "content": prompt + "\nPlease reason step by step, and put your final answer within \\boxed{}."},
        ]
  
        input_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )   
        if not args.deepseek:
            full_text = input_str + "<think>\n" + output  
        else:
            full_text = input_str + output 
        
        full_tokens = tokenizer(full_text, truncation=True, max_length=args.max_seq_length)
        input_ids = full_tokens['input_ids']
        labels = [-100] * len(input_ids)
        
        response_tokens = tokenizer(response_template, add_special_tokens=False)['input_ids']
        response_start = None
        for i in range(len(input_ids) - len(response_tokens) + 1):
            if input_ids[i:i+len(response_tokens)] == response_tokens:
                response_start = i + len(response_tokens)
                break
            
        assert response_start is not None
        
        if not args.mask or (not args.apply_all and len(input_ids) < 4000):
            if response_start is not None:
                labels[response_start:] = input_ids[response_start:]       
        else:
            num += 1
            parts = re.split(pattern, output)
            segment = [parts[0]]  
            for j in range(1, len(parts), 2):
                segment.append(parts[j] + parts[j + 1])  
                
            # add_ids = [0, len(segment)-1]  
            add_ids = [0, len(segment)-2, len(segment)-1]            
            segment_id = sorted(list(set(add_ids + segment_id)))
                
            if response_start is not None:
                for each_id in segment_id:
                    cur_seg = segment[each_id]
                    pre_segs = "".join(segment[:each_id])
                    pre_segs_tokens = tokenizer(pre_segs, add_special_tokens=False)['input_ids']
                    cur_seg_tokens = tokenizer(cur_seg, add_special_tokens=False)['input_ids']
                    target_start = response_start + len(pre_segs_tokens)
                    target_end = target_start + len(cur_seg_tokens)
                    if response_start + len(pre_segs_tokens) + len(cur_seg_tokens) <= args.max_seq_length:
                        labels[target_start:target_end] = input_ids[target_start:target_end]
                    else:
                        labels[target_start:] = input_ids[target_start:]
                        break

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    print("totoal num", num, len(input_ids_list))
    return {
        "input_ids": input_ids_list,
        "labels": labels_list
    }
if "json" in args.data_names:
    dataset = load_dataset("json", data_files=args.data_names)['train']
else:
    dataset = load_dataset(args.data_names, split = "train")

dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=["question", "solution", "answer", "selected_spans_ids", "segments"],  
    load_from_cache_file=False,
)
print(len(dataset), dataset[0].keys())
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    tokenizer = tokenizer,
    dataset_num_proc=2,
    packing = False,
    args = SFTConfig(
        max_seq_length = args.max_seq_length,
        remove_unused_columns = False,
        dataset_kwargs = {"skip_prepare_dataset": True},
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 1,
        warmup_ratio = 0.05,
        num_train_epochs = args.epochs, 
        learning_rate = args.learning_rate,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        lr_scheduler_type = "cosine",
        output_dir = args.output_dir,
        optim = "adamw_8bit", 
        seed = 3407,
        report_to = "wandb", 
        run_name = args.run_name,
        save_strategy = "epoch",
        overwrite_output_dir=True,
        save_total_limit = 3,
        save_only_model=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0
    ),
    callbacks=[EarlyStopAtEpochCallback()],
)
trainer.train()