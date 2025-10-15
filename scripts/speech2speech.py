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
import warnings

import omegaconf  # type: ignore
import torch  # type: ignore
import torchaudio  # type: ignore

warnings.filterwarnings("ignore")

# Fix OmegaConf validation: convert floats to ints in old checkpoints
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
        Plugins.instance().register(PkgSearchPathPlugin)
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
                    [
                        "encoder.pos_conv.0.weight_g",
                        "encoder.pos_conv.0.weight_v",
                    ]
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

# Tacotron combinations
configs = [
    ("tacotron", "hubert-base-ls960", "kmeans", 50),
    ("tacotron", "hubert-base-ls960", "kmeans", 100),
    ("tacotron", "hubert-base-ls960", "kmeans", 200),
    ("tacotron", "cpc-big-ll6k", "kmeans", 50),
    ("tacotron", "cpc-big-ll6k", "kmeans", 100),
    ("tacotron", "cpc-big-ll6k", "kmeans", 200),
    ("hifigan", "mhubert-base-25hz", "kmeans", 500),
    ("hifigan", "hubert-base-ls960-layer-9", "kmeans", 500),
    ("hifigan", "hubert-base-ls960-layer-9", "kmeans-expresso", 2000),
    ("hifigan", "mhubert-base-vp_mls_cv_8lang", "kmeans", 2000),
    ("hifigan", "mhubert-base-vp_mls_cv_8lang", "kmeans-expresso", 2000),
]

# Test Tacotron combinations
for vocoder_name, dense_model, quantizer, vocab_size in configs:
    encoder = SpeechEncoder.by_name(
        dense_model_name=dense_model,
        quantizer_model_name=quantizer,
        vocab_size=vocab_size,
        deduplicate=True,
    ).cuda()

    if vocoder_name == "tacotron":
        vocoder = TacotronVocoder.by_name(
            dense_model,
            quantizer,
            vocab_size,
        ).cuda()
    elif vocoder_name == "hifigan":
        vocoder = CodeHiFiGANVocoder.by_name(
            dense_model,
            quantizer,
            vocab_size,
        ).cuda()
    else:
        raise ValueError(f"Unknown vocoder: {vocoder_name}")

    vocod = vocoder_name[:4]
    model = dense_model.split("-")[0]
    config = f"{vocod}_{model}_{quantizer}_{vocab_size}"
    print(f"Testing: {config}")

    base = "/store/projects/lexical-benchmark/audio/symlinks/50h/"
    files = [
        base + "05/1087_LibriVox_en_seq_058.wav",  # 120s
        base + "05/1087_LibriVox_en_seq_010.wav",  # 30s
        base + "05/1087_LibriVox_en_seq_000.wav",  # 10s
        base + "05/1087_LibriVox_en_seq_002.wav",  # 5s
    ]
    for file in files:
        waveform, sr = torchaudio.load(file)
        encoded = encoder(waveform.cuda())
        units = encoded["units"]

        audio = vocoder(units)
        torchaudio.save(
            f"output/{config}_{file.split('/')[-1]}",
            audio.cpu().float().unsqueeze(0),
            vocoder.output_sample_rate,
        )
