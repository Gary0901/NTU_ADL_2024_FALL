import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from peft import PeftModel
from utils import get_prompt, get_bnb_config
import argparse
import os

def perplexity(
    model, tokenizer, data, max_length=2048,
):
    # [保持原有的 perplexity 函數不變]
    data_size = len(data)
    instructions = [get_prompt(x["instruction"]) for x in data]
    outputs = [x["output"] for x in data]

    # Tokenize data
    tokenized_instructions = tokenizer(instructions, add_special_tokens=False)
    tokenized_outputs = tokenizer(outputs, add_special_tokens=False)
    output_masks = []

    # Format data
    for i in range(data_size):
        instruction_input_ids = [tokenizer.bos_token_id] + \
            tokenized_instructions["input_ids"][i]
        output_input_ids = tokenized_outputs["input_ids"][i] + \
            [tokenizer.eos_token_id]
        tokenized_instructions["input_ids"][i] = instruction_input_ids + \
            output_input_ids
        tokenized_instructions["attention_mask"][i] = [
            1] * len(tokenized_instructions["input_ids"][i])
        output_mask = [0] * len(instruction_input_ids) + \
            [1] * len(output_input_ids)

        tokenized_instructions["input_ids"][i] = torch.tensor(
            tokenized_instructions["input_ids"][i][:max_length])
        tokenized_instructions["attention_mask"][i] = torch.tensor(
            tokenized_instructions["attention_mask"][i][:max_length])
        output_mask = torch.tensor(output_mask[:max_length])
        output_masks.append(output_mask)

    # Calculate ppl
    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for i in tqdm(range(data_size)):
        input_ids = tokenized_instructions["input_ids"][i].unsqueeze(0)
        attn_mask = tokenized_instructions["attention_mask"][i].unsqueeze(0)
        output_mask = output_masks[i].unsqueeze(0)
        label = input_ids

        with torch.no_grad():
            out_logits = model(input_ids, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_label = label[..., 1:].contiguous()
        shift_output_mask = output_mask[..., 1:].contiguous()
        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2),
             shift_label) * shift_output_mask).sum(1)
            / shift_output_mask.sum(1)
        )
        ppls += perplexity_batch.tolist()
    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9)."
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        required=True,
        help="Directory containing the checkpoints to evaluate"
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
        required=True,
        help="Path to test data."
    )
    args = parser.parse_args()

    # Load base model
    bnb_config = get_bnb_config()

    if args.base_model_path:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    else:
        model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"
        revision = "5073b2bbc1aa5519acdc865e99832857ef47f7c9"
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load test data
    with open(args.test_data_path, "r") as f:
        test_data = json.load(f)

    # 評估所有檢查點
    results = {}
    
    # 獲取所有檢查點路徑
    checkpoint_dirs = []
    if os.path.exists(os.path.join(args.checkpoints_dir, "final")):
        checkpoint_dirs.append(os.path.join(args.checkpoints_dir, "final"))
    
    for item in os.listdir(args.checkpoints_dir):
        if item.startswith("checkpoint-"):
            checkpoint_dirs.append(os.path.join(args.checkpoints_dir, item))
    
    # 按檢查點編號排序
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]) if "checkpoint-" in x else float("inf"))

    print("\n開始評估所有檢查點...")
    for checkpoint_path in checkpoint_dirs:
        print(f"\n評估檢查點: {checkpoint_path}")
        
        # 載入 LoRA 模型
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model.eval()
        
        # 計算 perplexity
        ppl = perplexity(model, tokenizer, test_data)
        results[checkpoint_path] = ppl["mean_perplexity"]
        
        print(f"Mean perplexity: {ppl['mean_perplexity']:.4f}")
        
        # 釋放記憶體
        del model
        torch.cuda.empty_cache()

    # 找出最佳檢查點
    best_checkpoint = min(results.items(), key=lambda x: x[1])
    print("\n評估結果總結:")
    for checkpoint, ppl in sorted(results.items(), key=lambda x: x[1]):
        print(f"{checkpoint}: {ppl:.4f}")
    print(f"\n最佳檢查點: {best_checkpoint[0]} (perplexity: {best_checkpoint[1]:.4f})")