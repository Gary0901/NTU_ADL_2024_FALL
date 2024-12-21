# inference_zero_shot.py
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import json
from utils import get_prompt_zero_shot, get_bnb_config
import argparse

def batch_inference(model, tokenizer,data,batch_size):
    """ 批量處理推理"""
    output_list = []
    for i in tqdm(range(0,len(data),batch_size)):
        
        batch = data[i:i+batch_size]
        prompts = [get_prompt_zero_shot(item["instruction"]) for item in batch]
        
        inputs = tokenizer(prompts,return_tensors="pt",padding=True)
        
        inputs = {k:v.to(model.device) for k,v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,        # 減少token數以加快速度
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_beams=1,              # 使用greedy search
                #early_stopping=True       # 提前停止生成
            )
        #批量解碼
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # 處理每個輸出
        for item, text in zip(batch, texts):
            output = text.split("專業古文學者:")[-1].strip()
            output_list.append({
                "id": item["id"],
                "instruction": item["instruction"],  # 保留原始instruction
                "output": output    # 只有output是模型生成的
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
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=get_bnb_config(),
        device_map="auto",torch_dtype=torch.float16,  # 使用半精度
        low_cpu_mem_usage=True
    )
    tokenizer.padding_side = 'left'

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set random seed
    set_seed(42)

    # Load test data
    with open(args.test_data_path, "r") as f:
        data = json.load(f)

    # Run inference
    model.eval()
    BATCH_SIZE = 8 
    output_list= batch_inference(model,tokenizer,data,BATCH_SIZE)

    # Save results
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_list, f, ensure_ascii=False,indent=2)

if __name__ == "__main__":
    main()
    
""" 
python inference_zero_shot.py \
    --test_data_path path/to/test.json \
    --output_path path/to/output/zero_shot_results.json
"""