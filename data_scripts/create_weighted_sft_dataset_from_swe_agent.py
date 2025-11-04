import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
from transformers import AutoTokenizer
import glob

IGNORE_INDEX = -100


def process_example(tokenizer, conversation_list: List[Dict[str, Any]]):
    """Process conversation and create input_ids, labels, and custom loss_mask."""
    if not isinstance(conversation_list, list):
        raise ValueError(f"The sample must be a list but got {type(conversation_list)}")

    input_ids = []
    labels = []
    loss_mask = []

    # Tokenize message-by-message using the chat template
    for i, m in enumerate(conversation_list):
        # Extract fields
        role = m.get("role", "")
        content = m.get("content", "")
        thought = m.get("thought", "")
        action = m.get("action", "")

        
        # Create sub-dictionary with only role and content
        msg_dict = {"role": role, "content": content}
        
        seg_ids = tokenizer.apply_chat_template(
            [msg_dict], tokenize=True, add_generation_prompt=False
        )

        # print (f"seg ids: {seg_ids}")
        input_ids.extend(seg_ids)
        
        if role != "assistant":
            # Non-assistant messages
            labels.extend([IGNORE_INDEX] * len(seg_ids))
            loss_mask.extend([0] * len(seg_ids))
        else:
            # Assistant messages
            labels.extend(seg_ids)
            
            # Check if action contains "str_replace_editor str_replace"
            has_str_replace = action and 'str_replace_editor str_replace' in action
            
            if thought:
                # Create thought lookup tokens
                thought_dict = {"role": role, "content": thought}
                
                thought_seg_ids = tokenizer.apply_chat_template(
                    [thought_dict], tokenize=True, add_generation_prompt=False
                )
                
                # Remove special tokens at beginning and end. 
                # 3 beginning tokens: <im_start> assistant \n 
                # 2 ending tokens: <im_end> \n
                thought_seg_ids = thought_seg_ids[3:-2]                
                
                # Find thought tokens in original seg_ids
                thought_mask = [0] * len(seg_ids)
                if thought_seg_ids:
                    # Look for the thought sequence in the original tokens
                    for j in range(len(seg_ids) - len(thought_seg_ids) + 1):
                        if seg_ids[j:j+len(thought_seg_ids)] == thought_seg_ids:
                            # Mark these positions as thought tokens
                            for k in range(len(thought_seg_ids)):
                                thought_mask[j + k] = 1
                            print (f"Found thought seg ids with offset 0")
                            break
                        elif seg_ids[j:j+len(thought_seg_ids) - 1] == thought_seg_ids[:-1]:
                            # Mark these positions as thought tokens
                            for k in range(len(thought_seg_ids)-1):
                                thought_mask[j + k] = 1
                            print (f"Found thought seg ids with offset -1")
                            break
                        elif seg_ids[j:j+len(thought_seg_ids) - 2] == thought_seg_ids[:-2]:
                            # Mark these positions as thought tokens
                            for k in range(len(thought_seg_ids)-2):
                                thought_mask[j + k] = 1
                            print (f"Found thought seg ids with offset -2")
                            break
                
                # Set loss masks
                for is_thought in thought_mask:
                    if is_thought:
                        loss_mask.append(1)  # Thought tokens get weight 1
                    else:
                        if has_str_replace:
                            loss_mask.append(4)  # Non-thought with str_replace gets weight 4
                        else:
                            loss_mask.append(2)  # Regular non-thought gets weight 2
            else:
                # No thought content
                if has_str_replace:
                    loss_mask.extend([4] * len(seg_ids))
                else:
                    loss_mask.extend([2] * len(seg_ids))

    assert len(input_ids) == len(labels) == len(loss_mask)

    return input_ids, labels, loss_mask


def extract_trajectory_query(traj_file: str) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Extract query from trajectory file and return query and action info."""
    with open(traj_file, 'r') as f:
        data = json.load(f)
    
    trajectory = data.get('trajectory', [])
    if not trajectory:
        raise ValueError(f"No trajectory found in {traj_file}")
    
    # Try to get query from last trajectory item
    query = trajectory[-1].get('query', [])
    
    # If query is empty, try second to last
    if not query and len(trajectory) > 1:
        query = trajectory[-2].get('query', [])
        
        if not query:
            raise ValueError(f"Empty query in both last and second-to-last trajectory items in {traj_file}")
    
    # Query is already a list of dicts with 'role' and 'content'
    
    return query


def main():
    parser = argparse.ArgumentParser(description='Create weighted SFT dataset from SWE-Agent logs')
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='/home/shared/swe-agent_logs/shramana/20251103_121356_openai/surya_1000_bugs',
        help='Input directory containing SWE-Agent logs'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=False,
        help='HuggingFace model path for tokenizer (required unless --no-tokenize is used)'
    )
    parser.add_argument(
        '--no-tokenize',
        action='store_true',
        help='Output raw messages without tokenization'
    )
    parser.add_argument(
        '--filter-file',
        type=str,
        default='/home/shared/swe-agent_logs/saurav/20251009_190531_openai/Qwen3/filtered_instances.txt',
        help='Text file containing bug names to filter (one per line)'
    )
    
    args = parser.parse_args()
    
    # Setup output file path
    if args.output_file is None:
        input_path = Path(args.input_dir)
        args.output_file = str(input_path.parent / f"{input_path.name}_weighted.jsonl")
    
    # Load tokenizer only if not in no-tokenize mode
    tokenizer = None
    if not args.no_tokenize:
        if not args.model_path:
            parser.error("--model_path is required unless --no-tokenize is specified")
        print(f"Loading tokenizer from: {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load filter list if provided
    filter_bugs = None
    if args.filter_file and os.path.exists(args.filter_file):
        print(f"Loading filter file: {args.filter_file}")
        with open(args.filter_file, 'r') as f:
            filter_bugs = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(filter_bugs)} bugs to process from filter file")
    
    print(f"Processing input directory: {args.input_dir}")
    print(f"Writing output to: {args.output_file}")
    
    # Process the dataset
    with open(args.output_file, 'w', encoding='utf-8') as outfile:
        
        processed_count = 0
        error_count = 0
        
        # Scan subdirectories
        for subdir in os.listdir(args.input_dir):
            subdir_path = os.path.join(args.input_dir, subdir)
            
            if not os.path.isdir(subdir_path):
                continue
            
            # Skip if filter is active and bug is not in the filter list
            if filter_bugs is not None and subdir not in filter_bugs:
                continue
            
            # Find trajectory files
            traj_files = glob.glob(os.path.join(subdir_path, '*traj'))
            
            if not traj_files:
                print(f"No trajectory files found in {subdir}")
                continue
            
            # Process the first trajectory file found
            traj_file = traj_files[0]
            print (f"Processing trajectory file: {traj_file}")
            
            try:
                # Extract query from trajectory
                messages = extract_trajectory_query(traj_file)
                
                if args.no_tokenize:
                    # Output raw messages without tokenization
                    # Filter to only include role and content fields
                    clean_messages = []
                    for msg in messages:
                        clean_msg = {
                            "role": msg.get("role", ""),
                            "content": msg.get("content", "")
                        }
                        clean_messages.append(clean_msg)
                    
                    output_entry = {
                        'messages': clean_messages
                    }
                else:
                    # Process the conversation with tokenization
                    input_ids, labels, loss_mask = process_example(tokenizer, messages)
                    
                    # Create output entry
                    output_entry = {
                        'messages': {
                            'input_ids': input_ids,
                            'labels': labels,
                            'loss_mask': loss_mask
                        }
                    }
                
                # Write to output file
                outfile.write(json.dumps(output_entry) + '\n')
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} examples...")                
            except Exception as e:
                error_count += 1
                print(f"Error processing {subdir}/{os.path.basename(traj_file)}: {e}")
                continue
    
    print(f"\nProcessing complete!")
    print(f"Total bugs processed: {processed_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Output written to: {args.output_file}")


if __name__ == "__main__":
    main()