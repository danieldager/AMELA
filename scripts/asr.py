#!/usr/bin/env python3
"""
ASR Pipeline using NVIDIA Canary-Qwen-2.5B for English transcription.

Each task writes its transcriptions to a separate file.
Merge all task outputs afterward with a simple script.

Usage:
    python asr.py --manifest data.json --task-id 0 --num-tasks 16

Input/Output format (JSONL):
    {"audio_filepath": "/path/to/expresso/audio/file.wav", "duration": 21.4}
    
After processing, adds 'text' field:
    {"audio_filepath": "...", "duration": 21.4, "text": "Transcription here."}
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
import time

warnings.filterwarnings('ignore')

import torch # type: ignore
from nemo.collections.speechlm2.models import SALM # type: ignore


def process_manifest(
    manifest_path: str,
    model_name: str,
    batch_size: int,
    max_tokens: int,
    device: str,
    task_id: int = 0,
    num_tasks: int = 1,
):
    """Process audio files and write transcriptions to task-specific output file."""
    
    # Read manifest
    with open(manifest_path, 'r') as f:
        all_entries = [json.loads(line) for line in f]
    
    # Filter entries for this task
    task_entries = []
    for i, entry in enumerate(all_entries):
        if i % num_tasks == task_id:
            # Only process if 'text' field is missing or empty
            if 'text' not in entry or not entry['text']:
                task_entries.append(entry)
    
    if num_tasks > 1:
        print(f"Task {task_id}/{num_tasks}: Processing {len(task_entries)} files")
    else:
        print(f"Processing {len(task_entries)} files")
    
    if len(task_entries) == 0:
        print("No files to process (all already transcribed)")
        return
    
    # Load model once
    print(f"Loading model: {model_name}")
    model = SALM.from_pretrained(model_name)
    if device == "cuda":
        model = model.cuda()
    model.eval()
    print("Model loaded")
    
    # Process files
    transcriptions = {}  # audio_filepath -> text
    processed_count = 0
    total_duration = 0.0
    start_time = time.time()
    
    # Process in batches
    for batch_start in range(0, len(task_entries), batch_size):
        batch_entries = task_entries[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(task_entries) + batch_size - 1) // batch_size
        
        print(f"Batch {batch_num}/{total_batches} ({len(batch_entries)} files)")
        
        # Transcribe batch
        try:
            audio_paths = [e["audio_filepath"] for e in batch_entries]
            
            # Prepare prompts
            prompts = []
            for audio_path in audio_paths:
                prompts.append([{
                    "role": "user",
                    "content": f"Transcribe the following: {model.audio_locator_tag}",
                    "audio": [audio_path]
                }])
            
            # Generate transcriptions
            with torch.no_grad():
                answer_ids = model.generate(
                    prompts=prompts,
                    max_new_tokens=max_tokens,
                )
            
            # Store transcriptions
            for entry, ids in zip(batch_entries, answer_ids):
                text = model.tokenizer.ids_to_text(ids.cpu())
                transcriptions[entry["audio_filepath"]] = text
                
                processed_count += 1
                total_duration += entry.get('duration', 0)
            
            print(f"Batch {batch_num} complete: {len(batch_entries)} files transcribed")
            
        except Exception as e:
            print(f"ERROR: Batch {batch_num} failed: {e}", file=sys.stderr)
            continue
    
    # Write task-specific output file
    manifest_path_obj = Path(manifest_path)
    output_dir = manifest_path_obj.parent / f".{manifest_path_obj.stem}_transcriptions"
    output_dir.mkdir(exist_ok=True)
    task_output = output_dir / f"task_{task_id:04d}.json"
    
    print(f"Writing task output to {task_output}")
    with open(task_output, 'w') as f:
        json.dump(transcriptions, f, indent=2)
    
    # Summary
    elapsed = time.time() - start_time
    rtf = total_duration / elapsed if elapsed > 0 else 0
    
    print("=" * 60)
    print(f"Task {task_id} completed: {processed_count}/{len(task_entries)} files")
    print(f"Total audio: {total_duration:.1f}s")
    print(f"Processing time: {elapsed:.1f}s")
    print(f"RTF: {rtf:.1f}x")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="ASR pipeline with array job support - writes per-task output files"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to JSONL manifest with audio_filepath column"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/canary-qwen-2.5b",
        help="Model name (default: nvidia/canary-qwen-2.5b)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens per transcription (default: 256)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: 'cuda' or 'cpu' (default: auto)"
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=0,
        help="Task ID for array job (0-indexed)"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=1,
        help="Total number of parallel tasks"
    )
    
    args = parser.parse_args()
    
    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU", file=sys.stderr)
        args.device = "cpu"
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
    try:
        process_manifest(
            args.manifest,
            args.model,
            args.batch_size,
            args.max_tokens,
            args.device,
            args.task_id,
            args.num_tasks,
        )
    except Exception as e:
        print(f"ERROR: Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
