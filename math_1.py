import json 
import os 
import pandas as pd
import torch
import tensorflow_datasets as tsdf 
import pandas 
from transformers import (
    AutoModelForCausalLM,
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
        for file in folder_path:
            file_path = os.path.join(folder_path, file)
            data = json.load(file_path)
            problem.append(data.get('question'))
            level.append(data.get('level'))
            type.append(data.get('type'))
            solution.append(data.get('answer'))
    
    complete_data = {
        'question': problem,
        'level' : level,
        'type' : type,
        'answer': solution
    }
    df = pd.DataFrame(complete_data)
    return complete_data

def tf_math_datasets():
    data = tsdf.load('math_dataset')
    return data


def convert_dataset(data):
    question = data['question']
    answer = data['answer']
    prompt = f'<s>[INST]{question} [/INST]{answer}</s>'
    return prompt

if __name__ == '__main__' :
    path = '/Users/omkarnaik/Downloads/MATH'
    
    dataset = load_data(path=path)
    tf_dataset = tf_math_datasets()
    
    base_model = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    
    llama_tokenizer = AutoTokenizer.from_pretrained(base_model,trust_remote_code = True )
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = 'right'
    
    
    quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False)
    
    model = AutoModelForCausalLM.from_pretrained(base_model,
                                                quantization_config = quant_config,
                                                device_map = {'':0})
    model.config.use_cache = False 
    model.config.pretraining_tp = 1
    
    peft_parameters = LoraConfig(lora_alpha= 16,
                            lora_dropout= 0.1,
                            r = 8,
                            bias = 'none',
                            task_type='CASUAL_LM')
    
    train_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard")
    
    fine_tuning = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_params)
    
    fine_tuning.train()
