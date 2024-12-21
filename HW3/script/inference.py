import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json
from tqdm import tqdm
import argparse
import os
from typing import List, Dict
from utils import  get_bnb_config, get_prompt

def load_model_and_tokenizer(
    base_model_path: str,
    adapter_model_path: str,
    device: str = "cuda"
) -> tuple:
    """
    Load the base model and adapter model.
    """
    print(f"Loading base model from {base_model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        use_fast=True
    )
    
    # Load base model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=get_bnb_config(),
        device_map="auto" if device == "cuda" else "cpu",
        trust_remote_code=True
    )
    
    print(f"Loading adapter from {adapter_model_path}...")
    model = PeftModel.from_pretrained(
        model,
        adapter_model_path,
        is_trainable=False
    )
    
    return model, tokenizer

def generate_response(
    instruction: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_type: str = "default",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_beams: int = 3,
    **kwargs
) -> str:
    """
    Generate a response for a given instruction.
    """
    # Select prompt type
    prompt = get_prompt(instruction)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response
    response = response.replace(prompt, "").strip()
    return response

def process_dataset(
    input_file: str,
    output_file: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_type: str = "default",
    **generate_kwargs
) -> None:
    """
    Process a dataset and save the results.
    """
    # Load the test data
    print(f"Loading test data from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test examples")
    print("First example format:", test_data[0].keys())
    
    results = []
    for item in tqdm(test_data, desc="Generating responses"):
        try:
            instruction = item['instruction']
            original_output = item.get('output', '')
            
            response = generate_response(
                instruction,
                model,
                tokenizer,
                prompt_type=prompt_type,
                **generate_kwargs
            )
            
            # Format output
            result = {
                'id': item.get('id', ''),
                'output': response.strip(),
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing instruction: {instruction}")
            print(f"Error message: {str(e)}")
            results.append({
                'id': item.get('id', ''),
                'output': f"Error: {str(e)}",
            })
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Print some statistics
    print("\nGeneration Results Summary:")
    print(f"Total examples processed: {len(results)}")
    errors = len([r for r in results if r['output'].startswith('Error:')])
    print(f"Successful generations: {len(results) - errors}")
    print(f"Errors: {errors}")

def main():
    parser = argparse.ArgumentParser(description="Generate responses using a fine-tuned model")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--adapter_model", type=str, required=True, help="Path to adapter model (LoRA weights)")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save results")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--num_beams", type=int, default=3, help="Number of beams for beam search")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    
    args = parser.parse_args()
    
    # Check if CUDA is available when device is set to cuda
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        args.device = "cpu"
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.base_model,
        args.adapter_model,
        args.device
    )
    
    # Process dataset
    process_dataset(
        args.input_file,
        args.output_file,
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams
    )

if __name__ == "__main__":
    main()