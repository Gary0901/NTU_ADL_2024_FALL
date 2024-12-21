import json
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from tqdm import tqdm
import os
import argparse

# 加載模型和分詞器
model_path = "./model_result/beam_search_4/checkpoint-7230"
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
model = MT5ForConditionalGeneration.from_pretrained(model_path)

# 將模型移到GPU(如果可用)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

def generate_summary(text, strategy='beam_search', **kwargs):
    # 準備輸入
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 設置生成參數
    gen_kwargs = {
        'max_length': 128,
        'early_stopping': True,
        **kwargs
    }
    
    if strategy == 'greedy':
        gen_kwargs['num_beams'] = 1
        gen_kwargs['early_stopping'] = False  # 修改這裡
    elif strategy == 'beam_search':
        gen_kwargs['num_beams'] = 20
        gen_kwargs['no_repeat_ngram_size'] = 2
    elif strategy == 'top_k':
        gen_kwargs['do_sample'] = True
        gen_kwargs['top_k'] = 50
        """ gen_kwargs['temperature'] = 0.7 """
    elif strategy == 'top_p':
        gen_kwargs['do_sample'] = True
        gen_kwargs['top_p'] = 0.92
        """ gen_kwargs['temperature'] = 0.7 """
    
    # 生成摘要
    summary_ids = model.generate(inputs["input_ids"], **gen_kwargs)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

def main(input_file, output_file, strategy):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile, total=total_lines, desc="Processing articles"):
            data = json.loads(line)
            
            maintext = data['maintext']
            
            # 生成摘要
            generated_title = generate_summary(maintext, strategy=strategy)
            
            # 創建輸出字典並寫入文件
            output = {
                "id": data['id'],
                "title": generated_title
            }
            json.dump(output, outfile, ensure_ascii=False)
            outfile.write('\n')
    
    print(f"處理完成。結果已保存到 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summaries using different strategies")
    parser.add_argument("--input", default="./data/public.jsonl", help="Input file path")
    parser.add_argument("--output", default="./output/output.jsonl", help="Output file path")
    parser.add_argument("--strategy", choices=['greedy', 'beam_search', 'top_k', 'top_p'], default='beam_search', help="Decoding strategy")
    
    args = parser.parse_args()
    
    main(args.input, args.output, args.strategy)