#!/usr/bin/env python3
"""
Synthesize audio from token files using HiFi-GAN vocoder.

Usage:
    python synthesize.py --input_dir output/generations/lstm_h2048_r0.0003_e1024_l3_b64_d0.1/chunk0 (--overwrite)
"""

import argparse
from pathlib import Path

import torch
import torchaudio  # type: ignore

from textless.vocoders.hifigan.vocoder import CodeHiFiGANVocoder  # type: ignore
from utils import should_skip_existing


def synthesize_tokens(tokens, vocoder, device="cuda"):
    """Synthesize audio from tokens using HiFi-GAN."""
    # Ensure 1D
    if tokens.ndim > 1:
        tokens = tokens.squeeze()

    # Strip SOS token if present at index 0
    if len(tokens) > 0 and tokens[0] == 2000:
        tokens = tokens[1:]

    # Convert to int32 and move to device
    tokens = tokens.to(dtype=torch.int32, device=device)

    with torch.no_grad():
        waveform = vocoder(code=tokens)

    return waveform


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize audio from token files using HiFi-GAN"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory with .pt token files (searched recursively)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .wav files (default: skip)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Expect tokens in tokens/ subdirectory
    tokens_dir = input_dir if input_dir.name == "tokens" else input_dir / "tokens"
    if not tokens_dir.exists():
        raise FileNotFoundError(f"Tokens directory not found: {tokens_dir}")
    
    # Create speech/ subdirectory as sibling to tokens/
    speech_dir = tokens_dir.parent / "speech"
    speech_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(tokens_dir.rglob("*.pt"))
    if len(pt_files) == 0:
        print("No .pt files found. Exiting.")
        return

    # Filter files based on overwrite flag
    files_to_process = []
    for pt_file in pt_files:
        # Compute relative path and corresponding wav file in speech/ dir
        rel_path = pt_file.relative_to(tokens_dir)
        wav_file = speech_dir / rel_path.with_suffix(".wav")
        
        if not should_skip_existing(wav_file, args.overwrite):
            files_to_process.append((pt_file, wav_file))

    print(f"Tokens: {tokens_dir}")
    print(f"Speech: {speech_dir}")
    print(f"Found: {len(pt_files)} .pt files | Processing: {len(files_to_process)}\n")

    if len(files_to_process) == 0:
        print("All files already synthesized. Use --overwrite to regenerate.")
        return

    print("Loading HiFi-GAN vocoder...")
    vocoder = CodeHiFiGANVocoder.by_name(
        "mhubert-base-vp_mls_cv_8lang",
        "kmeans-expresso",
        2000,
    ).cuda()
    vocoder.eval()
    print(f"Vocoder loaded\n")

    print("Synthesizing audio...")
    print("-" * 60)

    success_count = 0
    error_count = 0

    for i, (pt_file, wav_file) in enumerate(files_to_process, 1):
        try:
            tokens = torch.load(pt_file)

            # Check for internal SOS token
            if tokens.ndim == 1 and len(tokens) > 1 and 2000 in tokens[1:].tolist():
                print(f"  WARNING: {pt_file.name} has internal SOS token")

            # Synthesize and save
            waveform = synthesize_tokens(tokens, vocoder, device=args.device)

            # Ensure waveform is 2D [channels, samples] for torchaudio.save
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # Add channel dimension

            # Create parent directory if needed (for nested structures)
            wav_file.parent.mkdir(parents=True, exist_ok=True)
            
            torchaudio.save(wav_file, waveform.cpu(), sample_rate=16000)

            success_count += 1

            if i % 10 == 0 or i == len(files_to_process):
                print(f"  Processed {i}/{len(files_to_process)}")

        except Exception as e:
            print(f"  ERROR processing {pt_file.name}: {e}")
            error_count += 1

    print("-" * 60)
    print(
        f"\nSynthesized {success_count} files"
        + (f" ({error_count} errors)" if error_count > 0 else "")
    )


if __name__ == "__main__":
    main()
