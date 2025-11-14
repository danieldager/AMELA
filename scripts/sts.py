"""
Speech-to-Speech Resynthesis Script

This script tests various vocoder and encoder combinations for speech resynthesis.

COMPATIBILITY FIXES FOR OLD FAIRSEQ CHECKPOINTS:
------------------------------------------------
This script includes several monkey-patches to ensure compatibility between:
- Old fairseq checkpoints (commit dd106d9)
- Modern PyTorch (2.6+)
- OmegaConf 2.0.6 / Hydra 1.0.7

The main issues addressed:
1. OmegaConf Validation: Old checkpoints store integers as floats (e.g., 50.0 instead of 50).
   We patch OmegaConf.merge to auto-convert floats to ints before validation.

2. Weight Normalization: Old checkpoints have incompatible weight_g/weight_v shapes for
   pos_conv layers. We remove these keys and let the model reinitialize them.
   See also: textlesslib/textless/data/hubert_feature_reader.py (modified to use strict=False)

3. PyTorch Security: PyTorch 2.6+ requires allowlisting classes for safe deserialization.

For detailed documentation, see: scripts/README.md
"""

import argparse
import json
import csv
import warnings
from pathlib import Path
from datetime import datetime

import omegaconf  # type: ignore
import torch  # type: ignore
import torchaudio  # type: ignore
import torchaudio.transforms as T  # type: ignore

warnings.filterwarnings("ignore")

# OmegaConf patch: convert floats to ints
_original_merge = omegaconf.OmegaConf.merge


def _patched_merge(*configs):
    def fix_floats(obj):
        if isinstance(obj, dict):
            return {k: fix_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [fix_floats(v) for v in obj]
        elif isinstance(obj, float) and obj.is_integer():
            return int(obj)
        return obj

    fixed = []
    for cfg in configs:
        try:
            if hasattr(cfg, "_metadata"):
                container = omegaconf.OmegaConf.to_container(cfg)
                fixed.append(omegaconf.OmegaConf.create(fix_floats(container)))
            elif isinstance(cfg, dict):
                fixed.append(fix_floats(cfg))
            else:
                fixed.append(cfg)
        except:
            fixed.append(cfg)
    return _original_merge(*fixed)


omegaconf.OmegaConf.merge = _patched_merge

# Register Hydra pkg:// source to avoid errors with old checkpoints
try:
    from hydra.core.config_search_path import ConfigSearchPath  # type: ignore
    from hydra.core.plugins import Plugins  # type: ignore
    from hydra.plugins.search_path_plugin import SearchPathPlugin  # type: ignore

    class PkgSearchPathPlugin(SearchPathPlugin):
        def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
            # Just register pkg:// as a valid scheme - we don't actually use it
            pass

    # Try to register (may fail silently if already registered or not needed)
    try:
        Plugins.instance().register(PkgSearchPathPlugin)  # type: ignore
    except:
        pass
except:
    pass

import fairseq.checkpoint_utils  # type: ignore
import fairseq.data.dictionary  # type: ignore
from textless.data.speech_encoder import SpeechEncoder  # type: ignore
from textless.vocoders.hifigan.vocoder import CodeHiFiGANVocoder  # type: ignore
from textless.vocoders.tacotron2.vocoder import TacotronVocoder  # type: ignore

# Allowlist fairseq classes for PyTorch 2.6+ security
torch.serialization.add_safe_globals(
    [
        argparse.Namespace,
        fairseq.data.dictionary.Dictionary,
    ]
)

# Patch checkpoint loading to remove incompatible weight normalization keys
_original_load_checkpoint = fairseq.checkpoint_utils.load_checkpoint_to_cpu


def _patched_load_checkpoint(path, *args, **kwargs):
    state = _original_load_checkpoint(path, *args, **kwargs)

    # Remove incompatible pos_conv weight_norm keys from old checkpoints
    if "model" in state:
        model_state = state["model"]
        keys_to_remove = []

        # Remove old weight_g/weight_v if they have incompatible shapes
        if "encoder.pos_conv.0.weight_g" in model_state:
            weight_g = model_state["encoder.pos_conv.0.weight_g"]
            if weight_g.dim() == 3 and weight_g.shape[0] != 1:
                keys_to_remove.extend(
                    ["encoder.pos_conv.0.weight_g", "encoder.pos_conv.0.weight_v"]
                )

        # Remove old BatchNorm-like format if it exists
        if "encoder.pos_conv.0.weight" in model_state:
            keys_to_remove.extend(
                [
                    "encoder.pos_conv.0.weight",
                    "encoder.pos_conv.0.running_mean",
                    "encoder.pos_conv.0.running_var",
                    "encoder.pos_conv.0.num_batches_tracked",
                    "encoder.pos_conv.1.weight",
                    "encoder.pos_conv.1.bias",
                ]
            )

        for key in keys_to_remove:
            model_state.pop(key, None)

    return state


fairseq.checkpoint_utils.load_checkpoint_to_cpu = _patched_load_checkpoint


def load_manifest(manifest_path):
    """Load manifest from CSV or JSONL file."""
    manifest_path = Path(manifest_path)

    if manifest_path.suffix == ".jsonl" or manifest_path.suffix == ".json":
        # JSONL format
        with open(manifest_path, "r") as f:
            entries = [json.loads(line) for line in f]
    elif manifest_path.suffix == ".csv":
        # CSV format
        entries = []
        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(row)
    else:
        raise ValueError(
            f"Unsupported manifest format: {manifest_path.suffix}. Use .csv or .jsonl"
        )

    return entries


def process_manifest(
    manifest_path: str, dataset_name: str, task_id: int = 0, num_tasks: int = 1
):
    """Process all audio files from manifest."""

    # Load manifest
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] Reading manifest: {manifest_path}")
    entries = load_manifest(manifest_path)
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] Loaded {len(entries)} entries")

    # Filter for this task (array job parallelization)
    if num_tasks > 1:
        entries = [e for i, e in enumerate(entries) if i % num_tasks == task_id]
        print(f"Task {task_id}/{num_tasks}: Processing {len(entries)} files")

    # Load encoder and vocoder
    dense_model, quantizer, vocab_size = (
        "mhubert-base-vp_mls_cv_8lang",
        "kmeans-expresso",
        2000,
    )

    encoder = SpeechEncoder.by_name(
        dense_model_name=dense_model,
        quantizer_model_name=quantizer,
        vocab_size=vocab_size,
        deduplicate=True,
    ).cuda()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Encoder loaded")

    vocoder = CodeHiFiGANVocoder.by_name(dense_model, quantizer, vocab_size).cuda()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Vocoder loaded")

    SAMPLE_RATE = 16000
    processed_count = 0

    for entry in entries:
        input_path = entry["audio_filepath"]

        # Use just the filename (original path is in the CSV)
        filename = Path(input_path).name
        
        # Output path: flat structure under dataset name
        output_path = Path("output") / dataset_name / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            print(f"Skipping (exists): {output_path}")
            continue

        print(f"Processing [{processed_count + 1}/{len(entries)}]: {input_path}")

        try:
            # Load and preprocess audio
            waveform, sr = torchaudio.load(input_path)

            # Ensure mono
            if waveform.shape[0] > 1:
                print(f"  Warning: {waveform.shape[0]} channels, using first")
                waveform = waveform[0:1, :]

            # Resample to 16kHz
            if sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sr, new_freq=SAMPLE_RATE
                )

            # Encode: Audio → Discrete Units
            encoded = encoder(waveform.cuda())
            units = encoded["units"]
            
            # Debug: show units format for vocoder compatibility
            print(f"  Units shape: {units.shape}, dtype: {units.dtype}")
            print(f"  Units sample: {units[:10]}")
            
            audio = vocoder(units)

            # Save
            torchaudio.save(
                str(output_path),
                audio.cpu().float().unsqueeze(0),
                vocoder.output_sample_rate,
            )
            processed_count += 1

        except Exception as e:
            print(f"ERROR processing {input_path}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"Task {task_id} completed: {processed_count}/{len(entries)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Speech-to-speech resynthesis from manifest (CSV or JSONL)"
    )
    parser.add_argument(
        "--manifest", required=True, help="Path to manifest (.csv or .jsonl)"
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset name (e.g., 'expresso')"
    )
    parser.add_argument(
        "--task-id", type=int, default=0, help="Task ID for array jobs (0-indexed)"
    )
    parser.add_argument(
        "--num-tasks", type=int, default=1, help="Total number of parallel tasks"
    )

    args = parser.parse_args()
    process_manifest(args.manifest, args.dataset, args.task_id, args.num_tasks)
