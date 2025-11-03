import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any
import random
from transformers import AutoTokenizer

IGNORE_INDEX = -100

def process_example(tokenizer, conversation_list: Dict[str, Any]):        
    if not isinstance(conversation_list, list):
        raise ValueError(f"The sample must be a list but got {type(conversation_list)}")

    input_ids = []
    labels = []
    loss_mask = []

    # Tokenize message-by-message using the chat template so formatting stays consistent
    for i, m in enumerate(conversation_list):
        seg_ids = tokenizer.apply_chat_template(
            [m], tokenize=True, add_generation_prompt=False
        )
        input_ids.extend(seg_ids)
        if m["role"] == "assistant":
            labels.extend(seg_ids)
            loss_mask.extend([1] * len(seg_ids))
            # loss_mask.extend([float(random.randint(1,4))/4] * len(seg_ids))
        else:
            labels.extend([IGNORE_INDEX] * len(seg_ids))
            loss_mask.extend([0] * len(seg_ids))
            

    assert len(input_ids) == len(labels) == len(loss_mask)

    return input_ids, labels, loss_mask


def main():
    parser = argparse.ArgumentParser(description='Create weighted SFT dataset from JSONL file')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='/home/shared/megatron_dir/data/sft/train_data_sft_480b_375_swe_bench.jsonl',
        help='Input JSONL file path'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output JSONL file path (default: adds _weighted suffix to input filename)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='HuggingFace model path for tokenizer'
    )
    
    args = parser.parse_args()
    
    # Setup output file path
    if args.output_file is None:
        input_path = Path(args.input_file)
        output_path = input_path.parent / f"{input_path.stem}_weighted{input_path.suffix}"
        args.output_file = str(output_path)
    
    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    print(f"Processing input file: {args.input_file}")
    print(f"Writing output to: {args.output_file}")
    
    # Process the dataset
    with open(args.input_file, 'r', encoding='utf-8') as infile, \
         open(args.output_file, 'w', encoding='utf-8') as outfile:
        
        line_count = 0
        error_count = 0
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse JSON line
                data = json.loads(line.strip())
                
                # Extract messages from the input data
                if 'messages' not in data:
                    raise ValueError(f"Missing 'messages' key in line {line_num}")
                
                messages = data['messages']
                
                # Process the conversation
                input_ids, labels, loss_mask = process_example(tokenizer, messages)
                
                # Create output entry with messages key
                output_entry = {
                    'messages': {
                        'input_ids': input_ids,
                        'labels': labels,
                        'loss_mask': loss_mask
                    }
                }
                
                # Write to output file
                outfile.write(json.dumps(output_entry) + '\n')
                line_count += 1
                
                if line_count % 1000 == 0:
                    print(f"Processed {line_count} examples...")
                    
            except Exception as e:
                error_count += 1
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"\nProcessing complete!")
    print(f"Total lines processed: {line_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Output written to: {args.output_file}")


if __name__ == "__main__":
    main()