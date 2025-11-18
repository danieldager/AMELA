#!/usr/bin/env python3
"""
Generate audio token sequences from trained LSTM models.

Usage:
    python generate.py --model lstm_h2048_r0.0003_e1024_l3_b64_d0.1 --dataset chunk0 \
                       --num_samples 5 \
                       --temperatures 0.8,1.0,1.2 \
                       --top_k 50,100,None \
                       --top_p 0.9,0.95,None
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch

from models import load_model_from_name
from utils import timestamp_now


def parse_list_arg(arg_str, arg_type=float):
    """Parse comma-separated list, handling 'None' values."""
    values = []
    for val in arg_str.split(","):
        val = val.strip()
        values.append(None if val.lower() == "none" else arg_type(val))
    return values


def check_internal_sos(tokens, sos_token_id=2000):
    """Check if SOS token appears after position 0."""
    return len(tokens) > 1 and sos_token_id in tokens[1:].tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Generate audio token sequences from trained LSTM"
    )
    parser.add_argument(
        "--model", required=True, help="Model name (e.g., lstm_h2048_r0.0003_e1024_l3_b64_d0.1)"
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset name (e.g., chunk0, chunk0-23)"
    )
    parser.add_argument(
        "--checkpoint", "-cp", default=None, help="Checkpoint to load (default: best model)"
    )
    parser.add_argument(
        "--num_samples", "-ns", type=int, default=5, help="Samples per combination"
    )
    parser.add_argument(
        "--temperatures", "-temps", default="0.8,1.0,1.2", help="Temperatures (default: 0.8,1.0,1.2)"
    )
    parser.add_argument(
        "--top_k", default="50,100,None", help="Top-k values (default: 50,100,None)"
    )
    parser.add_argument(
        "--top_p", default="0.9,0.95,None", help="Top-p values (default: 0.9,0.95,None)"
    )
    parser.add_argument(
        "--max_length", type=int, default=500, help="Max tokens (default: 500, ~20s at 25Hz)"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"]
    )
    
    args = parser.parse_args()
    
    # Parse parameter lists
    temperatures = parse_list_arg(args.temperatures, float)
    top_k_values = parse_list_arg(args.top_k, int)
    top_p_values = parse_list_arg(args.top_p, float)
    
    print(f"Model: {args.model} | Dataset: {args.dataset}")
    print(f"Samples: {args.num_samples} per combo | Max length: {args.max_length}")
    print(f"Temps: {temperatures} | Top-k: {top_k_values} | Top-p: {top_p_values}")
    print(f"Total: {len(temperatures) * len(top_k_values) * len(top_p_values) * args.num_samples} samples")
    print("=" * 60)
    print()
    
    model = load_model_from_name(args.model, args.dataset, args.checkpoint, args.device)
    print(f"Model loaded\n")
    
    # Create output directory with tokens subdirectory
    output_dir = Path("output") / args.model / args.dataset / "tokens"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}\n")
    
    # Open CSV log
    csv_path = output_dir / "generation_log.csv"
    csv_exists = csv_path.exists()
    
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "sample_id",
            "temperature",
            "top_k",
            "top_p",
            "max_length",
            "actual_length",
            "has_internal_sos",
            "token_file",
            "timestamp",
        ],
    )
    
    if not csv_exists:
        csv_writer.writeheader()
    
    # Generate samples
    sample_count = 0
    
    print("Generating samples...")
    print("-" * 60)
    
    for temp in temperatures:
        for top_k in top_k_values:
            for top_p in top_p_values:
                # Convert None to values that disable the feature
                top_k_val = top_k if top_k is not None else 0
                top_p_val = top_p if top_p is not None else 1.0
                
                print(f"Params: temp={temp}, top_k={top_k}, top_p={top_p}")
                
                for sample_idx in range(args.num_samples):
                    sample_count += 1
                    
                    # Generate unique sample ID (5-digit zero-padded)
                    sample_id = f"{sample_count:05d}"
                    
                    # Create filename
                    top_k_str = f"topk{top_k}" if top_k is not None else "topkNone"
                    top_p_str = f"topp{top_p}" if top_p is not None else "toppNone"
                    filename = f"temp{temp}_{top_k_str}_{top_p_str}_{sample_id}.pt"
                    
                    # Start with SOS token
                    sos_token = torch.tensor([[2000]], dtype=torch.long, device=args.device)
                    
                    # Generate sequence
                    generated = model.generate(
                        inputs=sos_token,
                        max_length=args.max_length,
                        temperature=temp,
                        top_k=top_k_val,
                        top_p=top_p_val,
                        device=args.device,
                    )
                    
                    # Convert to CPU and squeeze
                    tokens = generated.squeeze().cpu()
                    
                    # Check for internal SOS token
                    has_internal_sos = check_internal_sos(tokens)
                    
                    if has_internal_sos:
                        print(f"  WARNING: Sample {sample_id} has internal SOS token!")
                    
                    # Save tokens
                    token_path = output_dir / filename
                    torch.save(tokens, token_path)
                    
                    # Log to CSV
                    csv_writer.writerow({
                        "sample_id": sample_id,
                        "temperature": temp,
                        "top_k": top_k if top_k is not None else "None",
                        "top_p": top_p if top_p is not None else "None",
                        "max_length": args.max_length,
                        "actual_length": len(tokens),
                        "has_internal_sos": has_internal_sos,
                        "token_file": filename,
                        "timestamp": timestamp_now("iso"),
                    })
                    
                    if (sample_idx + 1) % 5 == 0 or (sample_idx + 1) == args.num_samples:
                        print(f"  Generated {sample_idx + 1}/{args.num_samples} samples")
                
                print()
    
    csv_file.close()
    
    print("-" * 60)
    print(f"\nGenerated {sample_count} samples → {output_dir}")
    print(f"Completed: {timestamp_now('time')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
