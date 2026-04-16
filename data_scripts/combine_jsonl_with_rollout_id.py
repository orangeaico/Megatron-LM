#!/usr/bin/env python3
"""
Combine multiple JSONL files and add rollout_id field.

For each instance_id, entries from different files are assigned rollout_id
values (0, 1, 2, ...) in the order they appear. The output is sorted by
instance_id and rollout_id.

Usage:
    python combine_jsonl_with_rollout_id.py input1.jsonl input2.jsonl ... -o output.jsonl
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def combine_jsonl_files(input_files, output_file, exclude_incomplete=True):
    """
    Combine multiple JSONL files with rollout_id tracking.

    Args:
        input_files: List of input JSONL file paths
        output_file: Output JSONL file path
        exclude_incomplete: If True, exclude instance_ids not present in all input files
    """
    # Dictionary to track entries by instance_id
    # instance_data[instance_id] = [(entry, source_file_index), ...]
    instance_data = defaultdict(list)

    # Track which files each instance_id appears in
    instance_file_presence = defaultdict(set)

    # Read all input files
    num_files = len(input_files)
    for file_idx, input_file in enumerate(input_files):
        print(f"Reading {input_file}...")
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    if 'instance_id' not in entry:
                        print(f"Warning: Line {line_num} in {input_file} missing instance_id, skipping")
                        continue
                    instance_id = entry['instance_id']
                    instance_data[instance_id].append((entry, file_idx))
                    instance_file_presence[instance_id].add(file_idx)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num} in {input_file}: {e}")
                    continue

    # Filter instance_ids if exclude_incomplete is True
    if exclude_incomplete:
        complete_instance_ids = {
            instance_id for instance_id, files in instance_file_presence.items()
            if len(files) == num_files
        }
        excluded_count = len(instance_data) - len(complete_instance_ids)
        if excluded_count > 0:
            print(f"Excluding {excluded_count} instance_ids not present in all {num_files} input files")
        instance_data = {k: v for k, v in instance_data.items() if k in complete_instance_ids}

    # Process entries and assign rollout_ids
    all_entries = []
    for instance_id, entries in instance_data.items():
        # Group by instance_id and assign rollout_id based on order
        for rollout_id, (entry, _) in enumerate(entries):
            entry['rollout_id'] = rollout_id
            all_entries.append(entry)

    # Sort by instance_id and rollout_id
    all_entries.sort(key=lambda x: (x['instance_id'], x['rollout_id']))

    # Write output
    print(f"Writing {len(all_entries)} entries to {output_file}...")
    with open(output_file, 'w') as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"Done! Processed {len(instance_data)} unique instance_ids")


def main():
    parser = argparse.ArgumentParser(
        description='Combine multiple JSONL files and add rollout_id field'
    )
    parser.add_argument(
        'input_files',
        nargs='+',
        help='Input JSONL files to combine'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output JSONL file'
    )
    parser.add_argument(
        '--exclude-incomplete',
        action='store_true',
        default=True,
        help='Exclude instance_ids not present in all input files (default: True)'
    )
    parser.add_argument(
        '--include-incomplete',
        dest='exclude_incomplete',
        action='store_false',
        help='Include instance_ids even if not present in all input files'
    )

    args = parser.parse_args()

    # Validate input files exist
    for input_file in args.input_files:
        if not Path(input_file).exists():
            print(f"Error: Input file {input_file} not found")
            return 1

    combine_jsonl_files(args.input_files, args.output, args.exclude_incomplete)
    return 0


if __name__ == '__main__':
    exit(main())
