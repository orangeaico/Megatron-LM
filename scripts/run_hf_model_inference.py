#!/usr/bin/env python3
"""
Script to load a HuggingFace model from local path and run inference with a prompt.
"""

import torch
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_inference(model_path, prompt, max_new_tokens=100, temperature=1.0, top_p=1.0, do_sample=False):
    """
    Load model from local path and generate text from prompt.
    """
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist!")
        return
    
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Automatically use GPU if available
        trust_remote_code=True,
        local_files_only=True
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"\nModel loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Mode: {'Sampling (non-deterministic)' if do_sample else 'Greedy (deterministic)'}")
    
    # Tokenize input
    print(f"\nPrompt: {prompt}")
    print("-" * 80)
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\nGenerated text:")
    print("=" * 80)
    print(generated_text)
    print("=" * 80)
    
    # Show only the new tokens
    new_text = generated_text[len(prompt):].strip()
    print(f"\nNew tokens only:")
    print("-" * 80)
    print(new_text)
    print("-" * 80)
    
    # Show token count
    input_tokens = inputs['input_ids'].shape[1]
    output_tokens = outputs.shape[1]
    new_tokens = output_tokens - input_tokens
    
    print(f"\nToken counts:")
    print(f"  Input tokens: {input_tokens}")
    print(f"  Output tokens: {output_tokens}")
    print(f"  New tokens generated: {new_tokens}")

def main():
    parser = argparse.ArgumentParser(description="Run HuggingFace model inference from local path")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Local path to the model directory")
    parser.add_argument("--prompt", type=str, 
                        default="Once upon a time, in a land far away,",
                        help="Input prompt for text generation")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling (only used if --do-sample is set)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p (nucleus) sampling (only used if --do-sample is set)")
    parser.add_argument("--do-sample", action="store_true",
                        help="Use sampling instead of greedy decoding (non-deterministic)")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode for multiple prompts")
    
    args = parser.parse_args()
    
    if args.interactive:
        # Interactive mode
        print("Running in interactive mode. Type 'quit' or 'exit' to stop.\n")
        
        # Load model once
        if not os.path.exists(args.model_path):
            print(f"Error: Model path {args.model_path} does not exist!")
            return
        
        print(f"Loading model from: {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Model loaded! Ready for prompts.\n")
        
        while True:
            prompt = input("\nEnter prompt (or 'quit' to exit): ")
            if prompt.lower() in ['quit', 'exit']:
                break
            
            if not prompt.strip():
                continue
            
            # Generate
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            print("\nGenerating...")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=args.do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_text = generated_text[len(prompt):].strip()
            
            print("\nGenerated continuation:")
            print("-" * 80)
            print(new_text)
            print("-" * 80)
    else:
        # Single prompt mode
        run_inference(
            args.model_path, 
            args.prompt, 
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.do_sample
        )

if __name__ == "__main__":
    main()