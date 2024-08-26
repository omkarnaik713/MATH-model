import json 
import os 
import pandas as pd
import torch
import tensorflow_datasets as tsdf 
import pandas 
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

def load_data(path):
    
    problem = []
    level = []
    type = []
    solution = [] 
    
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path) :
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path,file)
                if os.path.isfile(file_path) and file.endswith('.json'):
                    with open(file_path,'r') as f :
                        data = json.load(f)
                    problem.append(data.get('question'))
                    level.append(data.get('level'))
                    type.append(data.get('type'))
                    solution.append(data.get('answer'))
    return problem, solution

def tf_math_datasets():
    data = tsdf.load('math_dataset')
    return data


def convert_dataset(question,answer):
    prompt = []
    for ques, ans in zip(question,answer) :
        formatted_data = f'<s>[INST]{ques} [/INST]{ans}</s>'
        prompt.append(formatted_data)
    return prompt

if __name__ == '__main__' :
    train_path = '/Users/omkarnaik/Downloads/MATH/train'
    
    test_path = '/Users/omkarnaik/Downloads/MATH/test'
    
    question, answer = load_data(path=train_path)
    converted_data = convert_dataset(question,answer)
    #tf_dataset = tf_math_datasets()
    
    base_model = 'NousResearch/Llama-2-7b-chat-hf'
    
    lora_r = 64
    lora_alpha = 16
    lora_dropout = 0.1
    
    use_4bit = True
    bnb_4bit_compute_dtype = 'float16'
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = False
    
    output_dir = "./results"
    
    num_train_epochs = 1
    
    fp16 = False 
    bf16 = False 
    
    per_device_train_batch_size = 4
    per_device_eval_batch_size = 4
    
    gradient_accumulation_steps = 1
    gradient_checkpointing = True 
    
    max_grad_norm = 0.3
    learning_rate = 2e-4
    weight_decay = 0.001
    optim = "paged_adamw_32bit"
    lr_scheduler_type = 'cosine'
    max_steps = -1
    warmup_ratio = 0.03
    group_by_length = True
    
    logging_steps = 25
    
    max_seq_length = None 
    packing = False
    
    ## loading the tokenizer 
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant
    )
    
    ## load the base model 
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config = bnb_config,
        device_map = 'auto'
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code = True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    ## loading LoRA Configuration 
    
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r = lora_r,
        bias = 'none',
        task_type='CAUSAL_LM',
    )
    
    ## setting training parameters 
    training_arguments = TrainingArguments(
        output_dir = output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim= optim,
        save_steps = 25,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to='tensorboard'
    )
    
    trainer = SFTTrainer(
        model = model,
        train_dataset = converted_data,
        peft_config=peft_config,
        dataset_text_field='text',
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args = training_arguments,
        packing = packing,
    )
    
    trainer.train()