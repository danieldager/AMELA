# AMELA - Acoustic Modeling for Early Language Acquisition

Speech processing pipelines for HPC/SLURM clusters: Speech-to-Speech resynthesis, Voice Activity Detection, and Automatic Speech Recognition.

## Pipelines

### 1. Speech-to-Speech (STS) - Audio Resynthesis
Uses textlesslib/fairseq with 2021 checkpoints. Requires Python 3.9.

```bash
conda create -n textless python=3.9 -y
conda activate textless
git clone git@github.com:facebookresearch/textlesslib.git
pip install -e textlesslib/
pip install git+https://github.com/pytorch/fairseq.git@dd106d9534b22e7db859a6b87ffd7780c38341f8
pip install 'omegaconf==2.0.6' 'hydra-core==1.0.7' 'librosa==0.9.2'
```

### 2. Voice Activity Detection (VAD) - Speech Segmentation
CPU-based multiprocessing. Requires Python 3.11+.

```bash
conda create -n vad python=3.11 -y
conda activate vad
pip install git+https://github.com/TEN-framework/ten-vad.git
```

### 3. Automatic Speech Recognition (ASR) - Transcription
Uses NVIDIA Canary-Qwen-2.5B. Requires Python 3.10+, GPU.

**Setup:**
```bash
conda create -n canary python=3.10 -y
conda activate canary
pip install "nemo_toolkit[all]"
```

**Usage:**
```bash
# Process with 30 GPUs
./scripts/submit_asr.sh metadata/your_data.json 30

# What happens:
#   1. Split job (CPU) - divides manifest into 30 splits
#   2. Process jobs (30 GPUs) - transcribe splits in parallel
#   3. Merge job (CPU) - combines results into output/your_data_results.json
```

## Data Format

All pipelines use JSONL (newline-delimited JSON):
```json
{"audio_filepath": "/absolute/path/file.wav", "duration": 21.387}
{"audio_filepath": "/absolute/path/file2.wav", "duration": 33.191}
```

**Always use absolute paths** - compute nodes may have different working directories.

## File Structure

```
scripts/            # Pipeline scripts and SLURM launchers
  submit_asr.sh     # Main ASR submission script
  asr_split.slurm   # Step 1: Split manifest (CPU)
  asr_process.slurm # Step 2: Process splits (GPU array)
  asr_merge.slurm   # Step 3: Merge results (CPU)
metadata/           # Input manifests (JSONL)
output/             # Results
logs/               # SLURM job logs
```

### Critical: Apply Compatibility Fix

After installation, modify one line in textlesslib:

**File**: `textlesslib/textless/data/hubert_feature_reader.py`  
**Line 32-34**: Add `strict=False` parameter

```python
# To this:
model, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [self.checkpoint_path],
    strict=False,  # Allow loading old checkpoints with incompatible weights
)
```

## Why These Specific Versions?

Old fairseq checkpoints (2021) have compatibility issues with modern libraries:

- **OmegaConf 2.0.x**: Has `II` interpolation (needed by fairseq) but not overly strict validation
- **Hydra 1.0.x**: Compatible with OmegaConf 2.0.x
- **Fairseq dd106d9**: The exact commit textlesslib was developed against

See **TECHNICAL.md** for detailed compatibility fixes and implementation patterns.


## References

- [Textlesslib](https://github.com/facebookresearch/textlesslib)
- [Fairseq](https://github.com/pytorch/fairseq)
- [TEN-VAD](https://github.com/TEN-framework/ten-vad)
- [NeMo](https://github.com/NVIDIA/NeMo)
