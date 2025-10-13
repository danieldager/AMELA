import argparse

import fairseq.data.dictionary  # type: ignore
import torch
import torchaudio  # type: ignore
from textless.data.speech_encoder import SpeechEncoder  # type: ignore
from textless.vocoders.hifigan.vocoder import CodeHifiGanVocoder  # type: ignore
from textless.vocoders.tacotron2.vocoder import TacotronVocoder  # type: ignore

# Allowlist fairseq classes for PyTorch 2.6+ security
torch.serialization.add_safe_globals(
    [
        argparse.Namespace,
        fairseq.data.dictionary.Dictionary,
    ]
)

dense_model_name = "hubert-base-ls960"
quantizer_model_name, vocab_size = "kmeans", 100

encoder = SpeechEncoder.by_name(
    dense_model_name=dense_model_name,
    quantizer_model_name=quantizer_model_name,
    vocab_size=vocab_size,
    deduplicate=True,
).cuda()

tacotron = TacotronVocoder.by_name(
    dense_model_name,
    quantizer_model_name,
    vocab_size,
).cuda()

hifigan = CodeHifiGanVocoder.by_name(
    dense_model_name,
    quantizer_model_name,
    vocab_size,
).cuda()

vocoders = {
    "tacotron": tacotron,
    "hifigan": hifigan,
}

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

    for name, vocoder in vocoders.items():
        audio = vocoder(units)
        torchaudio.save(
            f"output/{name}_{file.split('/')[-1]}",
            audio.cpu().float().unsqueeze(0),
            vocoder.output_sample_rate,
        )
