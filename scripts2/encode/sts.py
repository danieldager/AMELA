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

import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from functools import partial

import numpy as np
import omegaconf  # type: ignore
import torch  # type: ignore
import torchaudio  # type: ignore

from utils import load_manifest_rows

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

# ============================================================================
# Patching
# ============================================================================

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


# ============================================================================
# Shared Setup
# ============================================================================

SAMPLE_RATE = 16000
ENCODER = None
VOCODER = None


def load_models():
    """Load encoder and vocoder models (cached globally)."""
    global ENCODER, VOCODER
    if ENCODER is None:
        dense_model = "mhubert-base-vp_mls_cv_8lang"
        quantizer, vocab_size = "kmeans", 2000

        ENCODER = SpeechEncoder.by_name(
            dense_model_name=dense_model,
            quantizer_model_name=quantizer,
            vocab_size=vocab_size,
            deduplicate=True,
        ).cuda()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Encoder loaded")

        VOCODER = CodeHiFiGANVocoder.by_name(dense_model, quantizer, vocab_size).cuda()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Vocoder loaded")

    return ENCODER, VOCODER


def resynthesize(waveform: torch.Tensor, sr: int) -> np.ndarray:
    """Resynthesize audio through encoder-vocoder pipeline."""
    encoder, vocoder = load_models()

    # Ensure mono
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = waveform[0:1, :]
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    # Encode → Decode
    with torch.no_grad():
        units = encoder(waveform.cuda())["units"]
        audio = vocoder(units).cpu().numpy()

    return audio


# ============================================================================
# Manifest-based Processing
# ============================================================================


def process_manifest(manifest_path: str, output_name: str = None, task_id: int = 0, num_tasks: int = 1):
    """
    Process audio files from manifest (CSV or JSONL).

    Args:
        manifest_path: Path to input manifest
        output_name: Output folder name (default: extracted from manifest name)
        task_id: Task ID for parallel processing (0-indexed)
        num_tasks: Total number of parallel tasks
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Reading manifest: {manifest_path}")
    entries = load_manifest_rows(manifest_path)

    # Output folder name
    if output_name is None:
        output_name = Path(manifest_path).stem.split("_")[0]
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(entries)} entries → output/{output_name}/")

    # Filter for this task (array job parallelization)
    if num_tasks > 1:
        entries = [e for i, e in enumerate(entries) if i % num_tasks == task_id]
        print(f"Task {task_id + 1}/{num_tasks}: Processing {len(entries)} files")

    # Load models
    encoder, vocoder = load_models()

    count = 0
    log_every = max(1, len(entries) // 10)

    for i, entry in enumerate(entries):
        input_path = entry["audio_filepath"]
        filename = Path(input_path).name
        output_path = Path("output") / output_name / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            continue

        if i % log_every == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing [{i+1}/{len(entries)}]: {filename}")

        try:
            waveform, sr = torchaudio.load(input_path)
            audio = resynthesize(waveform, sr)
            torchaudio.save(str(output_path), torch.from_numpy(audio).unsqueeze(0), vocoder.output_sample_rate)
            count += 1
        except Exception as e:
            print(f"ERROR processing {input_path}: {e}")
            continue

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Task {task_id} completed: {count}/{len(entries)} files")


# ============================================================================
# CLI
# ============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Speech-to-speech resynthesis from manifest"
    )
    parser.add_argument("manifest", help="Path to manifest (.csv or .jsonl)")
    parser.add_argument("-o", "--output", help="Output folder name (default: auto)")
    parser.add_argument("--task-id", type=int, default=0, help="Task ID for array jobs")
    parser.add_argument("--num-tasks", type=int, default=1, help="Total parallel tasks")

    args = parser.parse_args()
    process_manifest(args.manifest, args.output, args.task_id, args.num_tasks)
