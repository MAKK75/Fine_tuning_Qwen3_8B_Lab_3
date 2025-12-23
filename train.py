import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
import numpy as np
import random
import transformers.activations
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

if not hasattr(transformers.activations, "PytorchGELUTanh"):
    transformers.activations.PytorchGELUTanh = transformers.activations.NewGELUActivation

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

gc.collect()
torch.cuda.empty_cache()
seed_everything(42)  

MODEL_ID = "NOVORDSEC/qwen3-8b-awq-int4"
NEW_MODEL_NAME = "qwen3-8b-awq-finetuned"
DATASET_NAME = "mlabonne/guanaco-llama2-1k"

def train():
    print(f"GPUs available: {torch.cuda.device_count()}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto", 
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    model.config.use_cache = False 
    model.gradient_checkpointing_enable() 
    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads() 

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,            
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"] 
    )
    
    model = get_peft_model(model, peft_config)
    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    dataset = load_dataset(DATASET_NAME, split="train")

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=1,   
        gradient_accumulation_steps=4,   
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        logging_steps=10,
        optim="paged_adamw_32bit",       
        lr_scheduler_type="constant",
        save_strategy="no",
        report_to="none",
        ddp_find_unused_parameters=False,
        gradient_checkpointing=False,
        seed=42,          
        data_seed=42      
    )

    print("\nStarting training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=256, 
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    trainer.train()

    print(f"\nSaving model to {NEW_MODEL_NAME}...")
    trainer.model.save_pretrained(NEW_MODEL_NAME)
    tokenizer.save_pretrained(NEW_MODEL_NAME)
    print("Training completed successfully.")

if __name__ == "__main__":
    train()