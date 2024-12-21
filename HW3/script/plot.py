import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from peft import PeftModel
from utils import get_prompt, get_bnb_config
import argparse
import matplotlib.pyplot as plt
import os
from glob import glob

def get_step_from_checkpoint(checkpoint_path):
    """Extract step number from checkpoint path"""
    try:
        return int(checkpoint_path.split('checkpoint-')[1].split('/')[0])
    except:
        print(f"Warning: Could not parse step number from {checkpoint_path}")
        return 0

def calculate_ppl_for_checkpoint(base_model_path, checkpoint_path, test_data, tokenizer, bnb_config):
    """Calculate perplexity for a specific checkpoint"""
    print(f"Processing checkpoint: {checkpoint_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    
    from ppl import perplexity  # Import the perplexity function
    ppl = perplexity(model, tokenizer, test_data)
    
    del model
    torch.cuda.empty_cache()
    
    return ppl["mean_perplexity"]

def plot_learning_curve(steps, ppls, output_path):
    """Plot and save the learning curve with specified style"""
    plt.figure(figsize=(10, 6))
    
    # Plot the curve with a specific style
    plt.plot(steps, ppls, 'b-', linewidth=2)
    
    # Customize the plot
    plt.title('Learning Curve on the Public Testing Set', 
             fontsize=14, pad=15)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Perplexity Score', fontsize=12)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Customize axis
    plt.xlim(left=0)  # Start from 0
    plt.ylim(bottom=min(ppls)*0.95, top=max(ppls)*1.05)  # Add some padding to y-axis
    
    # Save with high DPI and tight layout
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="zake7749/gemma-2-2b-it-chinese-kyara-dpo",
        help="Path to the base model"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing the checkpoint folders"
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
        default="learning_curve.png",
        help="Path to save the learning curve plot"
    )
    args = parser.parse_args()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load test data
    print("Loading test data...")
    with open(args.test_data_path, "r") as f:
        test_data = json.load(f)

    # Get all checkpoint directories
    checkpoint_pattern = os.path.join(args.checkpoint_dir, "checkpoint-*", "adapter_model")
    checkpoint_paths = glob(checkpoint_pattern)
    
    if not checkpoint_paths:
        raise ValueError(f"No checkpoints found in {args.checkpoint_dir}")
    
    print(f"Found {len(checkpoint_paths)} checkpoints")
    checkpoint_paths.sort(key=get_step_from_checkpoint)

    # Get bnb config
    bnb_config = get_bnb_config()

    # Calculate perplexity for each checkpoint
    steps = []
    ppls = []
    
    # Load previous results if they exist
    if os.path.exists('ppl_results.json'):
        with open('ppl_results.json', 'r') as f:
            previous_results = json.load(f)
            if len(previous_results['steps']) > 0:
                print("Loading previous results...")
                steps = previous_results['steps']
                ppls = previous_results['perplexities']
    
    # Calculate for remaining checkpoints
    for checkpoint_path in tqdm(checkpoint_paths, desc="Processing checkpoints"):
        step = get_step_from_checkpoint(checkpoint_path)
        
        # Skip if already calculated
        if step in steps:
            continue
            
        ppl = calculate_ppl_for_checkpoint(
            args.base_model_path,
            checkpoint_path,
            test_data,
            tokenizer,
            bnb_config
        )
        
        steps.append(step)
        ppls.append(ppl)
        
        # Sort results by steps
        sorted_indices = np.argsort(steps)
        steps = [steps[i] for i in sorted_indices]
        ppls = [ppls[i] for i in sorted_indices]
        
        # Save intermediate results
        results = {
            'steps': steps,
            'perplexities': ppls
        }
        with open('ppl_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Plot learning curve
    print("Plotting learning curve...")
    plot_learning_curve(steps, ppls, args.output_path)
    
    # Print final results
    print("\nFinal Results:")
    for step, ppl in zip(steps, ppls):
        print(f"Step {step}: PPL = {ppl:.4f}")

if __name__ == "__main__":
    main()