import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List
from transformers import AutoTokenizer
import glob
import sys

IGNORE_INDEX = -100

ACTION_DEFAULT_WEIGHT = 2
action_weight_dict = {
    "think": 1,
    "str_replace_editor str_replace": 4,
    "finish": 4
}

def get_action_weight(action: str) -> int:
    """Get weight for a given action."""
    for act, weight in action_weight_dict.items():
        if action.startswith(act):
            return weight
    return ACTION_DEFAULT_WEIGHT  # Default weight for other actions

def count_tokens(tokenizer, conversation_list: List[Dict[str, Any]]) -> int:
    """Count total tokens in a conversation without processing labels/loss_mask."""
    if not isinstance(conversation_list, list):
        raise ValueError(f"The sample must be a list but got {type(conversation_list)}")
    
    total_tokens = 0
    for m in conversation_list:
        msg_dict = {"role": m.get("role", ""), "content": m.get("content", "")}
        seg_ids = tokenizer.apply_chat_template(
            [msg_dict], tokenize=True, add_generation_prompt=False
        )
        total_tokens += len(seg_ids)
    
    return total_tokens


def process_example(tokenizer, conversation_list: List[Dict[str, Any]], stats: Dict[str, int]):
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
            
            if thought:
                stats['assistant_with_thought'] += 1
                
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
                thought_found = False
                if thought_seg_ids:
                    # Look for the thought sequence in the original tokens
                    for j in range(len(seg_ids) - len(thought_seg_ids) + 1):
                        if seg_ids[j:j+len(thought_seg_ids)] == thought_seg_ids:
                            # Mark these positions as thought tokens
                            for k in range(len(thought_seg_ids)):
                                thought_mask[j + k] = 1
                            thought_found = True
                            break
                        elif seg_ids[j:j+len(thought_seg_ids) - 1] == thought_seg_ids[:-1]:
                            # Mark these positions as thought tokens
                            for k in range(len(thought_seg_ids)-1):
                                thought_mask[j + k] = 1
                            thought_found = True
                            break
                        elif seg_ids[j:j+len(thought_seg_ids) - 2] == thought_seg_ids[:-2]:
                            # Mark these positions as thought tokens
                            for k in range(len(thought_seg_ids)-2):
                                thought_mask[j + k] = 1
                            thought_found = True
                            break
                
                if thought_found:
                    stats['thought_found'] += 1
                
                action_weight = get_action_weight(action)
                # Set loss masks
                for is_thought in thought_mask:
                    if is_thought:
                        loss_mask.append(1)  # Thought tokens get weight 1
                    else: 
                        loss_mask.append(action_weight)  # Non-thought tokens
            else:                
                loss_mask.extend([get_action_weight(action)] * len(seg_ids))                

    assert len(input_ids) == len(labels) == len(loss_mask)

    return input_ids, labels, loss_mask


def extract_trajectory_query(traj_file: str) -> List[Dict[str, str]]:
    """Extract messages from trajectory file using history field."""
    with open(traj_file, 'r') as f:
        data = json.load(f)
    
    history = data.get('history', [])
    if not history:
        raise ValueError(f"No history found in {traj_file}")
    
    # Find the last assistant message index
    last_assistant_idx = -1
    for i in range(len(history) - 1, -1, -1):
        if history[i].get('role') == 'assistant':
            last_assistant_idx = i
            break
    
    if last_assistant_idx == -1:
        raise ValueError(f"No assistant messages found in history in {traj_file}")
    
    # Return all messages up to and including the last assistant message
    messages = history[:last_assistant_idx + 1]
    
    return messages


def main():
    parser = argparse.ArgumentParser(description='Create weighted SFT dataset from SWE-Agent logs')
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='/home/shared/swe-agent_logs/saurav/20251009_190531_openai/Qwen3',
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
        default="Qwen/Qwen3-Coder-30B-A3B-Instruct",
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
        default=None,
        help='Text file containing bug names to filter (one per line)'
    )
    parser.add_argument(
        '--filter-len',
        type=int,
        default=64000,
        help='Maximum token length for filtering examples (default: 64000)'
    )
    
    args = parser.parse_args()
    
    # Setup output file path
    if args.output_file is None:
        input_path = Path(args.input_dir)
        args.output_file = str(input_path / f"sft_dataset.jsonl")

    if args.filter_file is None:
        input_path = Path(args.input_dir)
        args.filter_file = str(input_path / f"resolved_submitted.txt")
    
    print ("=== CONFIGURATION ===")
    print (f"Input Directory: {args.input_dir}")
    print (f"Output File: {args.output_file}")
    print (f"Filter File: {args.filter_file}")
    
    # Load tokenizer (needed for both modes now due to filtering)
    if not args.model_path:
        parser.error("--model_path is required for token counting")
    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load filter list if provided
    filter_bugs = None
    if args.filter_file and os.path.exists(args.filter_file):
        print(f"Loading filter file: {args.filter_file}")
        with open(args.filter_file, 'r') as f:
            filter_bugs = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(filter_bugs)} bugs to process from filter file")
    else:
        sys.exit(f"ERROR: Filter file not found: {args.filter_file}, exiting..")
    
    print(f"Processing input directory: {args.input_dir}")
    print(f"Writing output to: {args.output_file}")
    print(f"Filter length: {args.filter_len:,} tokens")
    
    # Create overflow output file path
    output_path = Path(args.output_file)
    overflow_file = output_path.parent / f"{output_path.stem}_{args.filter_len}+{output_path.suffix}"
    
    # Process the dataset
    with open(args.output_file, 'w', encoding='utf-8') as outfile, \
         open(overflow_file, 'w', encoding='utf-8') as overflow_outfile:
        
        processed_count = 0
        error_count = 0
        filtered_count = 0
        overflow_count = 0
        
        # Initialize statistics
        stats = {
            'total_input_ids': 0,
            'total_labels': 0,
            'total_loss_mask': 0,
            'loss_mask_0': 0,
            'loss_mask_1': 0,
            'loss_mask_2': 0,
            'loss_mask_4': 0,
            'assistant_with_thought': 0,
            'thought_found': 0
        }
        
        # Initialize length buckets for input_ids
        length_buckets = {
            '<64000': 0,
            '64000-70000': 0,
            '70000-80000': 0,
            '80000-90000': 0,
            '90000-98000': 0,
            '>=98000': 0
        }
        
        # Per-example statistics list
        example_stats = []
        
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
            
            try:
                # Extract query from trajectory
                messages = extract_trajectory_query(traj_file)
                
                # Count tokens for filtering
                token_count = count_tokens(tokenizer, messages)
                
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
                        'messages': clean_messages,
                        'instance_id': subdir
                    }
                    
                    # Print per-example stats
                    print(f"Bug: {subdir} - Token count: {token_count:,}")
                    
                    # Update length buckets
                    if token_count < 64000:
                        length_buckets['<64000'] += 1
                    elif token_count < 70000:
                        length_buckets['64000-70000'] += 1
                    elif token_count < 80000:
                        length_buckets['70000-80000'] += 1
                    elif token_count < 90000:
                        length_buckets['80000-90000'] += 1
                    elif token_count < 98000:
                        length_buckets['90000-98000'] += 1
                    else:
                        length_buckets['>=98000'] += 1
                    
                    # Track example stats
                    example_stats.append({
                        'bug_id': subdir,
                        'input_ids_length': token_count
                    })
                else:
                    # Process the conversation with tokenization
                    input_ids, labels, loss_mask = process_example(tokenizer, messages, stats)
                    
                    # Update statistics
                    stats['total_input_ids'] += len(input_ids)
                    stats['total_labels'] += len(labels)
                    stats['total_loss_mask'] += len(loss_mask)
                    
                    # Count loss mask values
                    for mask_val in loss_mask:
                        if mask_val == 0:
                            stats['loss_mask_0'] += 1
                        elif mask_val == 1:
                            stats['loss_mask_1'] += 1
                        elif mask_val == 2:
                            stats['loss_mask_2'] += 1
                        elif mask_val == 4:
                            stats['loss_mask_4'] += 1
                    
                    # Track per-example stats
                    example_length = len(input_ids)
                    token_count = example_length  # For tokenize mode, use actual input_ids length
                    example_stats.append({
                        'bug_id': subdir,
                        'input_ids_length': example_length
                    })
                    
                    # Update length buckets
                    if example_length < 64000:
                        length_buckets['<64000'] += 1
                    elif example_length < 70000:
                        length_buckets['64000-70000'] += 1
                    elif example_length < 80000:
                        length_buckets['70000-80000'] += 1
                    elif example_length < 90000:
                        length_buckets['80000-90000'] += 1
                    elif example_length < 98000:
                        length_buckets['90000-98000'] += 1
                    else:
                        length_buckets['>=98000'] += 1
                    
                    # Print per-example stats
                    print(f"Bug: {subdir} - Input IDs length: {example_length:,}")
                    
                    # Create output entry
                    output_entry = {
                        'messages': {
                            'input_ids': input_ids,
                            'labels': labels,
                            'loss_mask': loss_mask
                        },
                        'instance_id': subdir
                    }
                
                # Write to appropriate output file based on token count
                if token_count <= args.filter_len:
                    outfile.write(json.dumps(output_entry) + '\n')
                    filtered_count += 1
                else:
                    overflow_outfile.write(json.dumps(output_entry) + '\n')
                    overflow_count += 1
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} examples...")         
            except Exception as e:
                error_count += 1
                print(f"Error processing {subdir}/{os.path.basename(traj_file)}: {e}")
                continue
    
    print(f"\nProcessing complete!")
    print(f"Total bugs processed: {processed_count}")
    print(f"Bugs within filter length ({args.filter_len:,}): {filtered_count}")
    print(f"Bugs exceeding filter length: {overflow_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Output written to: {args.output_file}")
    if overflow_count > 0:
        print(f"Overflow output written to: {overflow_file}")
    
    # Print statistics
    if not args.no_tokenize:
        print("\n=== TOKENIZATION STATISTICS ===")
        print(f"Total input_ids: {stats['total_input_ids']:,}")
        print(f"Total labels: {stats['total_labels']:,}")
        print(f"Total loss_mask: {stats['total_loss_mask']:,}")
        
        print("\n=== LOSS MASK DISTRIBUTION ===")
        if stats['total_loss_mask'] > 0:
            print(f"Loss mask 0 (non-assistant): {stats['loss_mask_0']:,} ({stats['loss_mask_0']/stats['total_loss_mask']*100:.1f}%)")
            print(f"Loss mask 1 (thought): {stats['loss_mask_1']:,} ({stats['loss_mask_1']/stats['total_loss_mask']*100:.1f}%)")
            print(f"Loss mask 2 (assistant non-thought): {stats['loss_mask_2']:,} ({stats['loss_mask_2']/stats['total_loss_mask']*100:.1f}%)")
            print(f"Loss mask 4 (assistant with str_replace): {stats['loss_mask_4']:,} ({stats['loss_mask_4']/stats['total_loss_mask']*100:.1f}%)")
        
        print("\n=== THOUGHT STATISTICS ===")
        print(f"Assistant messages with non-empty thought: {stats['assistant_with_thought']:,}")
        if stats['assistant_with_thought'] > 0:
            print(f"Thoughts successfully found in tokens: {stats['thought_found']:,} ({stats['thought_found']/stats['assistant_with_thought']*100:.1f}%)")
        
    
    print("\n=== TOKEN LENGTH DISTRIBUTION ===")
    for bucket, count in length_buckets.items():
        percentage = (count / processed_count * 100) if processed_count > 0 else 0
        print(f"{bucket}: {count} ({percentage:.1f}%)")
    
    print("\n=== PER-EXAMPLE STATISTICS ===")
    if example_stats:
        # Sort by length for better visualization
        sorted_examples = sorted(example_stats, key=lambda x: x['input_ids_length'], reverse=True)
        
        # Show top 10 longest examples
        print("\nTop 10 longest examples:")
        for i, ex in enumerate(sorted_examples[:10], 1):
            print(f"{i}. Bug: {ex['bug_id']} - Length: {ex['input_ids_length']:,}")
        
        # Calculate average and median
        lengths = [ex['input_ids_length'] for ex in example_stats]
        avg_length = sum(lengths) / len(lengths)
        sorted_lengths = sorted(lengths)
        median_length = sorted_lengths[len(lengths) // 2]
        
        print(f"\nAverage token length: {avg_length:,.1f}")
        print(f"Median token length: {median_length:,}")
        print(f"Min token length: {min(lengths):,}")
        print(f"Max token length: {max(lengths):,}")


if __name__ == "__main__":
    main()