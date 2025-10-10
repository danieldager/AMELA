import argparse
import torch
import torchaudio
from textless.data.speech_encoder import SpeechEncoder
from textless.vocoders.tacotron2.vocoder import TacotronVocoder
import fairseq.data.dictionary

# Allowlist fairseq classes for PyTorch 2.6+ security
torch.serialization.add_safe_globals([
    argparse.Namespace,
    fairseq.data.dictionary.Dictionary,
])

dense_model_name = "hubert-base-ls960"
quantizer_model_name, vocab_size = "kmeans", 100

# Load audio file
file = "/store/projects/lexical-benchmark/audio/symlinks/50h/05/1087_LibriVox_en_seq_058.wav"
waveform, sr = torchaudio.load(file)
print(f"Waveform shape: {waveform.shape}, sample rate: {sr}")

# Encode
encoder = SpeechEncoder.by_name(
    dense_model_name=dense_model_name,
    quantizer_model_name=quantizer_model_name,
    vocab_size=vocab_size,
    deduplicate=True,
).cuda()

encoded = encoder(waveform.cuda())

for k, v in encoded.items():
    print(f"{k}: {v.shape}")

units = encoded["units"]
print(f"Units shape: {units.shape}, dtype: {units.dtype}")

# Decode
vocoder = TacotronVocoder.by_name(
    dense_model_name,
    quantizer_model_name,
    vocab_size,
).cuda()

audio = vocoder(units)

torchaudio.save(
    "reconstructed.wav", 
    audio.cpu().float().unsqueeze(0), 
    vocoder.output_sample_rate
)

print("Done! Saved to reconstructed.wav")
