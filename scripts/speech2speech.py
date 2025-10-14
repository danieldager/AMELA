import argparse

import omegaconf  # type: ignore
import torch  # type: ignore
import torchaudio  # type: ignore


# Fix omegaconf strict validation for old checkpoints
def _patched_validate(self, value):
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return _original_validate(self, value)


_original_validate = omegaconf.nodes.IntegerNode._validate_and_convert_impl
omegaconf.nodes.IntegerNode._validate_and_convert_impl = _patched_validate


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

# Tacotron combinations
configs = [
    # ("tacotron", "hubert-base-ls960", "kmeans", 50),
    # ("tacotron", "hubert-base-ls960", "kmeans", 100),
    # ("tacotron", "hubert-base-ls960", "kmeans", 200),
    # ("tacotron", "cpc-big-ll6k", "kmeans", 50),
    # ("tacotron", "cpc-big-ll6k", "kmeans", 100),
    # ("tacotron", "cpc-big-ll6k", "kmeans", 200),
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
