# AMELA - Acoustic Modeling for Early Language Acquisition

Speech processing pipelines for HPC/SLURM clusters.

---

## Conda Environments

| Environment | Python | Pipelines |
|-------------|--------|-----------|
| `vad` | 3.11+ | VAD |
| `sts` | 3.9 | STS, Encoding, Synthesis |
| `train` | 3.10+ | Training, Generation |
| `asr` | 3.10+ | ASR, ASR Evaluation |

### Setup

```bash
# VAD
conda create -n vad python=3.11 -y && conda activate vad
pip install git+https://github.com/TEN-framework/ten-vad.git pandas soundfile torch torchaudio numpy

# STS / Synthesis
conda create -n sts python=3.9 -y && conda activate sts
git clone https://github.com/facebookresearch/textlesslib.git
pip install -e textlesslib/
pip install git+https://github.com/pytorch/fairseq.git@dd106d9534b22e7db859a6b87ffd7780c38341f8
pip install 'omegaconf==2.0.6' 'hydra-core==1.0.7' h5py pandas==1.5.3

# Encoding
conda create -n textless python=3.9 -y && conda activate textless
# Same as STS environment

# Training / Generation
conda create -n train python=3.10 -y && conda activate train
pip install torch pandas numpy transformers accelerate wandb

# ASR / ASR Evaluation
conda create -n asr python=3.10 -y && conda activate asr
pip install "nemo_toolkit[all]" torch torchaudio transformers whisper-normalizer jiwer
```

**STS Critical Fix**: Edit `textlesslib/textless/data/hubert_feature_reader.py` line 32 - add `strict=False` to `load_model_ensemble_and_task()`. See `TECHNICAL.md`.

---

## Pipelines

### 1. VAD - Voice Activity Detection

```bash
sbatch scripts/vad.slurm /path/to/audio/
```

**Output**: `metadata/<dirname>_<date>.csv` (VAD stats), `metadata/<dirname>_<date>.json` (manifest)

### 2. STS - Speech-to-Speech Resynthesis

```bash
sbatch --array=0-15 scripts/sts.slurm metadata/input.json
```

**Input**: JSONL with `audio_filepath`, `duration`  
**Output**: `output/<dataset>/` - Resynthesized 16kHz audio

### 3. ASR - Speech Recognition

```bash
sbatch --array=0-15 scripts/asr.slurm metadata/input.json

# After all tasks complete:
python -c "from utils import merge_asr_task_outputs; merge_asr_task_outputs('metadata/input.json')"
```

**Input**: JSONL with `audio_filepath`, `duration`  
**Output**: Updates manifest with `text` field

### 4. Encoding - Audio to Tokens

```bash
sbatch --array=0-5 scripts/encode.slurm metadata/input.csv
```

**Input**: CSV with `audio_filepath`, `file_id`  
**Output**: `.pt` token files

### 5. Training - LSTM Language Model

```bash
# Single run
sbatch scripts/train.slurm metadata/manifest.csv output/tokens_dir

# Grid search
python scripts/grid.py --manifest metadata/manifest.csv --tokens_dir output/tokens_dir
```

**Output**: `checkpoints/lstm_*/` - Model checkpoints

### 6. Generation & Synthesis

```bash
# Generate tokens
sbatch scripts/generate.slurm <model_name> <dataset_name>

# Synthesize audio
sbatch scripts/synthesize.slurm output/generations/<model_name>
```

**Output**: `.pt` token files → `.wav` audio files

### 7. ASR Evaluation

```bash
sbatch scripts/asr_eval.slurm metadata/ls-clean.csv
```

**Input**: CSV with `file_id`, `audio_filepath`, `transcription`  
**Output**: Updates manifest with WER/CER columns, saves `output/<split>_global_metrics.csv`

---

## Data Formats

**CSV**: Standard with headers (`file_id`, `audio_filepath`, `transcription`, etc.)

**JSONL** (`.json`):
```json
{"audio_filepath": "/absolute/path/file.wav", "duration": 21.4}
```

---

## Troubleshooting

### STS: "OmegaConf validation error"
Already handled by monkey-patches in `sts.py`. See `TECHNICAL.md`.

### STS: "Weight normalization shape mismatch"
Edit `textlesslib/textless/data/hubert_feature_reader.py` - add `strict=False`.

### ASR: "Task X wrote but text field missing"
Rerun failed task: `sbatch --array=X scripts/asr.slurm manifest.json`

### VAD: "No WAV files found"
Check path and file extensions.

---

## References

- [Textlesslib](https://github.com/facebookresearch/textlesslib)
- [TEN-VAD](https://github.com/TEN-framework/ten-vad)
- [NeMo](https://github.com/NVIDIA/NeMo)
- [LibriSpeech](https://www.openslr.org/12)

See **TECHNICAL.md** for implementation details.
