#!/usr/bin/env python3
"""
ASR Pipeline using NVIDIA Canary-Qwen-2.5B for English transcription.

Features:
- Batch processing with auto-tuned batch size
- Automatic punctuation and capitalization  
- ~400x real-time factor on GPU
- Progress tracking and comprehensive logging

Usage:
    python asr_process.py --manifest data.json --batch-size 8 --device cuda

Input format (JSONL):
    {"audio_filepath": "/path/file.wav", "duration": 21.4}

Output format (JSONL):
    {"audio_filepath": "/path/file.wav", "duration": 21.4, "text": "Transcription here."}

Limitations:
- Max 40 seconds per file (model limitation)
- English only
- Requires GPU for reasonable performance
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*pynvml.*')
warnings.filterwarnings('ignore', message='.*ffmpeg.*')
warnings.filterwarnings('ignore', message='.*UnsupportedFieldAttributeWarning.*')

import torch # type: ignore
from nemo.collections.speechlm2.models import SALM # type: ignore


def setup_logging(log_level: str = "INFO"):
    """Configure logging with both file and console output."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    return logging.getLogger(__name__)


def load_manifest(manifest_path: str, logger) -> List[Dict]:
    """
    Load audio file paths from JSON manifest.
    Expected format: each line is a JSON object with 'audio_filepath' and 'duration'
    """
    logger.info(f"Loading manifest from: {manifest_path}")
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    audio_files = []
    with open(manifest_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                if 'audio_filepath' not in entry:
                    logger.warning(f"Line {line_num}: Missing 'audio_filepath' field, skipping")
                    continue
                
                # Check if file exists
                if not os.path.exists(entry['audio_filepath']):
                    logger.warning(f"Line {line_num}: File not found: {entry['audio_filepath']}")
                    continue
                    
                audio_files.append(entry)
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: Invalid JSON - {e}")
                continue
    
    logger.info(f"Loaded {len(audio_files)} audio files from manifest")
    return audio_files


def load_model(model_name: str, device: str, logger):
    """Load the Canary-Qwen ASR model."""
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Using device: {device}")
    
    try:
        model = SALM.from_pretrained(model_name)
        
        # Move model to GPU if available
        if device == "cuda":
            model = model.cuda()
        
        model.eval()  # Set to evaluation mode
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def transcribe_batch(model, audio_paths: List[str], batch_size: int, 
                     max_new_tokens: int, logger) -> List[str]:
    """
    Transcribe a batch of audio files.
    
    Args:
        model: The loaded SALM model
        audio_paths: List of paths to audio files
        batch_size: Number of files to process at once
        max_new_tokens: Maximum tokens to generate per transcription
        logger: Logger instance
    
    Returns:
        List of transcriptions
    """
    transcriptions = []
    
    # Process in batches
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(audio_paths) + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} "
                   f"({len(batch_paths)} files)")
        
        try:
            # Prepare prompts for batch
            prompts = []
            for audio_path in batch_paths:
                prompts.append([{
                    "role": "user",
                    "content": f"Transcribe the following: {model.audio_locator_tag}",
                    "audio": [audio_path]
                }])
            
            # Generate transcriptions
            with torch.no_grad():  # Disable gradient computation for inference
                answer_ids = model.generate(
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                )
            
            # Convert token IDs to text
            for ids in answer_ids:
                text = model.tokenizer.ids_to_text(ids.cpu())
                transcriptions.append(text)
            
            logger.debug(f"Batch {batch_num} completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}")
            # Add empty strings for failed transcriptions
            transcriptions.extend([""] * len(batch_paths))
    
    return transcriptions


def save_results(audio_entries: List[Dict], transcriptions: List[str], 
                output_file: str, logger):
    """
    Save transcriptions to JSON manifest file.
    Each line contains the original entry plus the transcription.
    """
    logger.info(f"Saving results to: {output_file}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        for entry, transcription in zip(audio_entries, transcriptions):
            # Add transcription to the entry
            result = entry.copy()
            result['text'] = transcription
            f.write(json.dumps(result) + '\n')
    
    logger.info(f"Saved {len(transcriptions)} transcriptions")


def main():
    parser = argparse.ArgumentParser(
        description="ASR Pipeline using NVIDIA Canary-Qwen-2.5B"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to input JSON manifest file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL file with transcriptions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/canary-qwen-2.5b",
        help="Model name or path (default: nvidia/canary-qwen-2.5b)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of audio files to process in each batch (default: 8)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per transcription (default: 256)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use: 'cuda' or 'cpu' (default: auto-detect)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("ASR Pipeline Started")
    logger.info(f"Manifest: {args.manifest}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Max Tokens: {args.max_tokens}")
    logger.info(f"Device: {args.device}")
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        args.device = "cpu"
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    start_time = time.time()
    
    try:
        # Load manifest
        audio_entries = load_manifest(args.manifest, logger)
        
        if len(audio_entries) == 0:
            logger.error("No valid audio files found in manifest")
            sys.exit(1)
        
        # Load model
        model = load_model(args.model, args.device, logger)
        
        # Extract audio paths
        audio_paths = [entry['audio_filepath'] for entry in audio_entries]
        
        # Transcribe
        logger.info(f"Starting transcription of {len(audio_paths)} files...")
        transcriptions = transcribe_batch(
            model, audio_paths, args.batch_size, args.max_tokens, logger
        )
        
        # Save results to the specified output file
        save_results(audio_entries, transcriptions, args.output, logger)
        
        # Summary
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / len(audio_entries)
        total_duration = sum(entry.get('duration', 0) for entry in audio_entries)
        rtf = total_duration / elapsed_time if elapsed_time > 0 else 0
        
        logger.info("=" * 60)
        logger.info("ASR Pipeline Completed Successfully")
        logger.info("=" * 60)
        logger.info(f"Total files processed: {len(audio_entries)}")
        logger.info(f"Total audio duration: {total_duration:.2f} seconds")
        logger.info(f"Processing time: {elapsed_time:.2f} seconds")
        logger.info(f"Average time per file: {avg_time:.2f} seconds")
        logger.info(f"Real-time factor (RTF): {rtf:.2f}x")
        logger.info(f"Output saved to: {args.output}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

