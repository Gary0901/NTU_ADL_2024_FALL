import argparse
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice, AutoConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained multiple choice model")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model files")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test.json file")
    parser.add_argument("--context_file", type=str, required=True, help="Path to the context.json file")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for testing")
    parser.add_argument("--pretrained_model", type=str, default="hfl/chinese-lert-base", help="Name or path of the pretrained model") #有改!
    return parser.parse_args()

def preprocess_function(examples, tokenizer, max_length, context_dict):
    questions = [example["question"] for example in examples]
    paragraphs = [example["paragraphs"] for example in examples]
    
    first_sentences = [[question] * 4 for question in questions]
    second_sentences = [[context_dict[p] for p in para] for para in paragraphs]
    
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    
    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

def main():
    args = parse_args()

    # 獲取腳本的目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 將相對路徑轉換為絕對路徑
    model_dir = os.path.abspath(os.path.join(script_dir, '..', args.model_dir))
    test_file = os.path.abspath(os.path.join(script_dir, '..', args.test_file))
    context_file = os.path.abspath(os.path.join(script_dir, '..', args.context_file))

    # 確保所有路徑都存在
    for path in [model_dir, test_file, context_file]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")

    # 加載配置
    config = AutoConfig.from_pretrained(model_dir)
    print(f"Loaded config from {model_dir}")
    print(f"Config name_or_path: {config.name_or_path}")

    # 加載模型
    model = AutoModelForMultipleChoice.from_pretrained(model_dir, config=config)
    print(f"Loaded model from {model_dir}")

    # 嘗試從配置加載分詞器，如果失敗則使用指定的預訓練模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.name_or_path)
        print(f"Loaded tokenizer from config.name_or_path: {config.name_or_path}")
    except Exception as e:
        print(f"Failed to load tokenizer from config.name_or_path: {e}")
        print(f"Attempting to load tokenizer from specified pretrained model: {args.pretrained_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        print(f"Loaded tokenizer from {args.pretrained_model}")


    # 加載context.json
    with open(context_file, "r", encoding="utf-8") as f:
        context_dict = json.load(f)

    # 加載和處理測試數據
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test examples")
    print(f"First test example: {test_data[0]}")

    processed_test_data = preprocess_function(test_data, tokenizer, args.max_seq_length, context_dict)

    # 創建 DataLoader
    test_dataset = [{"input_ids": torch.tensor(input_ids), 
                     "attention_mask": torch.tensor(attention_mask)} 
                    for input_ids, attention_mask in zip(processed_test_data["input_ids"], processed_test_data["attention_mask"])]
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    # 進行預測
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            outputs = model(**batch)
            predictions.extend(outputs.logits.argmax(dim=-1).tolist())

    # 將預測結果與原始數據結合
    for i, pred in enumerate(predictions):
        test_data[i]["predicted_relevant"] = test_data[i]["paragraphs"][pred]

    # 保存結果
    output_file = "model1_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"預測結果已保存到 {output_file}")

if __name__ == "__main__":
    main()