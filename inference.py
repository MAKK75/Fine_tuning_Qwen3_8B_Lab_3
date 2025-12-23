import torch
import gc
import os
import pandas as pd
import numpy as np
import random
import transformers.activations
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftModel
from datasets import load_dataset
from huggingface_hub import model_info

if not hasattr(transformers.activations, "PytorchGELUTanh"):
    transformers.activations.PytorchGELUTanh = transformers.activations.NewGELUActivation

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

gc.collect()
torch.cuda.empty_cache()
seed_everything(42)

BASE_MODEL_ID = "NOVORDSEC/qwen3-8b-awq-int4" 
ADAPTER_REPO_ID = "NOVORDSEC/qwen3-8b-awq-finetuned-adapters"
ORIG_MODEL_ID = "Qwen/Qwen3-8B"
ORIG_ACC = 0.7292 #Получен на первом этапе, здесь просто неохота было второй раз считать
FRACTION = 0.2

def get_remote_size_gb(repo_id):
    try:
        info = model_info(repo_id, files_metadata=True)
        size = sum(s.size for s in info.siblings if s.size and any(ext in s.rfilename for ext in ['.safetensors', '.bin', '.pt']))
        return size / (1024**3)
    except Exception as e:
        print(f"Error getting size for {repo_id}: {e}")
        return 0

def run_mmlu_benchmark(model, tokenizer, fraction=0.2):
    subjects = [
        'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 
        'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 
        'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 
        'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 
        'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 
        'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 
        'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 
        'high_school_physics', 'high_school_psychology', 'high_school_statistics', 
        'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 
        'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 
        'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 
        'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 
        'professional_law', 'professional_medicine', 'professional_psychology', 
        'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
    ]
    
    device = model.device
    choices = ["A", "B", "C", "D"]
    choice_tokens = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in choices]
    
    detailed_results = {}
    total_correct = 0
    total_questions = 0
    
    model.eval()
    
    for subject in tqdm(subjects, desc="MMLU Inference"):
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test")
            num_samples = max(1, int(len(dataset) * fraction))
            dataset = dataset.select(range(num_samples))
            
            sub_correct = 0
            for item in dataset:
                prompt = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
                prompt += f"{item['question']}\n"
                prompt += f"(A) {item['choices'][0]}\n(B) {item['choices'][1]}\n(C) {item['choices'][2]}\n(D) {item['choices'][3]}\n"
                prompt += "Answer:"
                
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                with torch.inference_mode():
                    logits = model(**inputs).logits[0, -1, :]
                    relevant_logits = logits[choice_tokens]
                    pred = torch.argmax(relevant_logits).item()
                    
                    if pred == item['answer']:
                        sub_correct += 1
            
            acc = sub_correct / num_samples
            detailed_results[subject] = acc
            total_correct += sub_correct
            total_questions += num_samples
        except Exception as e:
            print(f"Error in {subject}: {e}")
            
    return (total_correct / total_questions if total_questions > 0 else 0), detailed_results

if __name__ == "__main__":
    print("Calculating model sizes...")
    size_orig = get_remote_size_gb(ORIG_MODEL_ID)
    size_base = get_remote_size_gb(BASE_MODEL_ID)
    size_adapter = get_remote_size_gb(ADAPTER_REPO_ID)

    final_size = size_base + size_adapter
    ratio = size_orig / final_size if final_size > 0 else 0

    print(f"Original Size: {size_orig:.2f} GB")
    print(f"Base Size: {size_base:.2f} GB")
    print(f"Adapter Size: {size_adapter:.4f} GB")
    print(f"Total Final Size: {final_size:.2f} GB")
    print(f"Compression Ratio: {ratio:.2f}x")

    print("\nLoading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("Loading Adapters...")
    model = PeftModel.from_pretrained(model, ADAPTER_REPO_ID)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    print("\nRunning Benchmark...")
    final_acc, detailed_results = run_mmlu_benchmark(model, tokenizer, fraction=FRACTION)

    drop = (ORIG_ACC - final_acc) / ORIG_ACC
    score = ratio / (1 + drop)

    print("\n" + "="*40)
    print(f"ИТОГИ ЭТАПА 2 (Fine-Tuning):")
    print(f"Compression Ratio: {ratio:.4f}")
    print(f"Original Accuracy: {ORIG_ACC:.4f}")
    print(f"Final Accuracy: {final_acc:.4f}")
    print(f"Performance Drop: {drop*100:.2f}%")
    print(f"ФИНАЛЬНЫЙ SCORE: {score:.4f}")
    print("="*40)

    subjects = sorted(list(detailed_results.keys()))
    csv_data = []
    for s in subjects:
        csv_data.append({
            "Subject": s,
            "Finetuned_Acc": detailed_results.get(s, 0.0)
        })

    df_final = pd.DataFrame(csv_data)
    df_final.to_csv("mmlu_finetuned_detailed.csv", index=False)
    print(f"Детальный отчет сохранен в mmlu_finetuned_detailed.csv")