import argparse
import io
import sys
import warnings

# Silence deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Suppress ANTLR version warning by redirecting stderr temporarily
class SuppressANTLRWarning:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = io.StringIO()
        return self

    def __exit__(self, *args):
        sys.stderr = self._original_stderr


with SuppressANTLRWarning():
    import fairseq.data.dictionary  # type: ignore

import librosa.util  # type: ignore
import torch
import torchaudio  # type: ignore

# Fix librosa.util.pad_center compatibility issue BEFORE importing textless
_original_pad_center = librosa.util.pad_center


def _patched_pad_center(data, size, axis=-1, **kwargs):
    # librosa 0.10+ changed signature from (data, size) to (data, *, size)
    return _original_pad_center(data, size=size, axis=axis, **kwargs)


librosa.util.pad_center = _patched_pad_center

from textless.data.speech_encoder import SpeechEncoder  # type: ignore
from textless.vocoders.hifigan.vocoder import CodeHifiGanVocoder  # type: ignore
from textless.vocoders.tacotron2.vocoder import TacotronVocoder  # type: ignore

# Allowlist all fairseq classes that might be in checkpoints
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
