# inference_few_shot.py
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import json
from utils import get_prompt_few_shot, get_bnb_config
import argparse

def batch_inference(model, tokenizer, data, batch_size):
    """ 批量處理推理"""
    output_list = []
    
    for i in tqdm(range(0, len(data), batch_size)):
        # 取得當前批次的數據
        batch = data[i:i+batch_size]
        
        # 使用few-shot prompt
        prompts = [get_prompt_few_shot(item["instruction"]) for item in batch]
        
        # Tokenize輸入
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成回應
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,     # 減少token數以加快速度
                temperature=0.7,        # 控制生成的創造性
                top_p=0.9,             # nucleus sampling參數
                do_sample=True,        # 使用採樣而不是貪婪搜索
                num_beams=1,           # 使用greedy search
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 批量解碼
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # 處理每個輸出
        for item, text in zip(batch, texts):
            # 提取專業古文學者的回應部分
            output = text.split("專業古文學者:")[-1].strip()
            output_list.append({
                "id": item["id"],
                "instruction": item["instruction"],
                "output": output
            })
    
    return output_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="zake7749/gemma-2-2b-it-chinese-kyara-dpo",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="Path to test data"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to prediction result"
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=get_bnb_config(),
        device_map="auto",
        torch_dtype=torch.float16,  # 使用半精度
        low_cpu_mem_usage=True
    )
    
    # 設置padding
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set random seed
    set_seed(42)
    
    # Load test data
    print(f"Loading test data from {args.test_data_path}")
    with open(args.test_data_path, "r") as f:
        data = json.load(f)
    
    # Run inference
    print("Starting inference...")
    model.eval()
    BATCH_SIZE = 8  # 可以根據GPU記憶體大小調整
    
    output_list = batch_inference(model, tokenizer, data, BATCH_SIZE)
    
    # Save results
    print(f"Saving results to {args.output_path}")
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_list, f, ensure_ascii=False, indent=2)
    
    print("Inference completed!")

if __name__ == "__main__":
    main()