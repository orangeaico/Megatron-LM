#!/usr/bin/env python3
"""
Script to load a HuggingFace model from local path and run inference with a prompt.
"""

import torch
import argparse
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path, rope_scaling_type=None, rope_scaling_factor=None, original_max_position_embeddings=None):
    """
    Load model and tokenizer from local path.
    Returns: (model, tokenizer)
    """
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist!")
        return None, None
    
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    # Prepare model kwargs
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
        "local_files_only": True
    }
    
    # Add rope_scaling if specified
    if rope_scaling_type and rope_scaling_factor:
        rope_scaling = {
            "type": rope_scaling_type,
            "factor": rope_scaling_factor
        }
        if original_max_position_embeddings:
            rope_scaling["original_max_position_embeddings"] = original_max_position_embeddings
        model_kwargs["rope_scaling"] = rope_scaling
        # model_kwargs["max_position_embeddings"] = 65000
        print(f"Using RoPE scaling: {rope_scaling}")
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully!")
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, 
                  top_p=1.0, do_sample=False, show_stats=True, top_k=None):
    """
    Generate text using the model.
    Returns: (generated_text, new_text, stats_dict)
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    
    # Generate
    with torch.no_grad():
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id
        }
        if top_k is not None:
            generation_kwargs["top_k"] = top_k
        
        outputs = model.generate(**generation_kwargs)
        print ("Generation config:", model.generation_config)
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    new_text = generated_text.strip()
    
    # Calculate stats
    stats = {
        'input_tokens': inputs['input_ids'].shape[1],
        'output_tokens': outputs.shape[1],
        'new_tokens': outputs.shape[1] - inputs['input_ids'].shape[1]
    }
    
    return generated_text, new_text, stats


def run_single_inference(model_path, prompt, max_new_tokens=100, temperature=1.0, 
                        top_p=1.0, do_sample=False, rope_scaling_type=None, 
                        rope_scaling_factor=None, original_max_position_embeddings=None, top_k=None):
    """
    Run single inference with detailed output.
    """
    model, tokenizer = load_model(model_path, rope_scaling_type, rope_scaling_factor, original_max_position_embeddings)
    if model is None:
        return
    
    print(f"\nModel type: {type(model).__name__}")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Mode: {'Sampling (non-deterministic)' if do_sample else 'Greedy (deterministic)'}")
    
    print(f"\nPrompt: {prompt}")
    print("-" * 80)
    
    print("Generating...")
    generated_text, new_text, stats = generate_text(
        model, tokenizer, prompt, max_new_tokens, temperature, top_p, do_sample, top_k=top_k
    )
    
    print("\nGenerated text:")
    print("=" * 80)
    print(generated_text)
    print("=" * 80)
    
    print(f"\nNew tokens only:")
    print("-" * 80)
    print(new_text)
    print("-" * 80)
    
    print(f"\nToken counts:")
    print(f"  Input tokens: {stats['input_tokens']}")
    print(f"  Output tokens: {stats['output_tokens']}")
    print(f"  New tokens generated: {stats['new_tokens']}")


def run_interactive_mode(args):
    """
    Run in interactive mode for multiple prompts.
    """
    print("Running in interactive mode. Type 'quit' or 'exit' to stop.\n")
    
    model, tokenizer = load_model(args.model_path, args.rope_scaling_type, args.rope_scaling_factor, args.original_max_position_embeddings)
    if model is None:
        return
    
    print("Ready for prompts.\n")
    
    while True:
        prompt = input("\nEnter prompt (or 'quit' to exit): ")
        if prompt.lower() in ['quit', 'exit']:
            break
        
        if not prompt.strip():
            continue
        
        print("\nGenerating...")
        _, new_text, _ = generate_text(
            model, tokenizer, prompt, args.max_new_tokens, 
            args.temperature, args.top_p, args.do_sample, show_stats=False, top_k=args.top_k
        )
        
        print("\nGenerated continuation:")
        print("-" * 80)
        print(new_text)
        print("-" * 80)


def run_chat_mode(args):
    """
    Run in chat mode with conversation format.
    """
    print("Running in chat mode...\n")
    
    model, tokenizer = load_model(args.model_path, args.rope_scaling_type, args.rope_scaling_factor, args.original_max_position_embeddings)
    if model is None:
        return
    
    # Read chat prompt from file
    with open('scripts/inference_prompt.txt', 'r') as f:
        chat_data = json.load(f)
    
    conversations = chat_data.get('conversations', [])
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)

    with open('scripts/prompt.txt', 'w') as f:
        f.write(prompt)
    
    print("Chat prompt formatted:")
    print("-" * 80)
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print("-" * 80)
    
    print("\nGenerating assistant response...")
    _, assistant_response, stats = generate_text(
        model, tokenizer, prompt, args.max_new_tokens,
        args.temperature, args.top_p, args.do_sample, top_k=args.top_k
    )

    with open('scripts/inference_response.txt', 'w') as f:
        f.write(assistant_response)
    
    print("\nAssistant response:")
    print("=" * 80)
    print(assistant_response)
    print("=" * 80)
    
    print(f"\nToken counts:")
    print(f"  Input tokens: {stats['input_tokens']}")
    print(f"  Output tokens: {stats['output_tokens']}")
    print(f"  New tokens generated: {stats['new_tokens']}")


def main():
    parser = argparse.ArgumentParser(description="Run HuggingFace model inference from local path")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Local path to the model directory")
    parser.add_argument("--prompt", type=str, 
                        default="Once upon a time, in a land far away,",
                        help="Input prompt for text generation")
    parser.add_argument("--max-new-tokens", type=int, default=2000,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling (only used if --do-sample is set)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p (nucleus) sampling (only used if --do-sample is set)")
    parser.add_argument("--do-sample", action="store_true", default=False,
                        help="Use sampling instead of greedy decoding (non-deterministic)")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode for multiple prompts")
    parser.add_argument("--chat", action="store_true",
                        help="Run in chat mode with system, user and assistant messages")
    parser.add_argument("--rope-scaling-type", type=str, default=None,
                        choices=["linear", "dynamic", "yarn", "longrope"],
                        help="RoPE scaling type (e.g., linear, dynamic, yarn, longrope)")
    parser.add_argument("--rope-scaling-factor", type=float, default=None,
                        help="RoPE scaling factor (e.g., 2.0, 4.0, 8.0)")
    parser.add_argument("--original-max-position-embeddings", type=int, default=None,
                        help="Original max position embeddings for RoPE scaling")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k sampling parameter (e.g., 1 for greedy, 50 for diverse)")
    
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive_mode(args)
    elif args.chat:
        run_chat_mode(args)
    else:
        # Single prompt mode
        run_single_inference(
            args.model_path, 
            args.prompt, 
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.do_sample,
            args.rope_scaling_type,
            args.rope_scaling_factor,
            args.original_max_position_embeddings,
            args.top_k
        )


if __name__ == "__main__":
    main()