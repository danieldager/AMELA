import pandas as pd

import torchaudio
from textless.data.speech_encoder import SpeechEncoder

model_name = "hubert-base-ls960"
quantizer_name, vocab_size = "kmeans", 100

df = pd.read_csv("../output/vad_results_14-03-10.csv.csv")
long = df[df["duration"] > 60]

# grab "filename" of first long file
filename = long.iloc[0]["filename"]
