# AMELA - Acoustic Modeling for Early Language Acquisition

Three independent speech processing pipelines for HPC/SLURM clusters. Each has its own Conda environment due to incompatible dependencies.

---

## Quick Start

```bash
# 1. Choose your pipeline
conda activate vad     # Voice Activity Detection (CPU)
conda activate sts     # Speech-to-Speech resynthesis (GPU)
conda activate asr     # Speech Recognition (GPU)

# 2. Run it
sbatch scripts/vad.slurm /path/to/audio/
sbatch scripts/sts.slurm metadata/manifest.json dataset_name
sbatch scripts/asr.slurm metadata/manifest.json
```

---

## Pipeline 1: Voice Activity Detection (VAD)

**What it does**: Analyzes audio files to find speech/silence segments and duration statistics.

**Environment**: Python 3.11+, CPU-only, multiprocessing

### Setup

```bash
conda create -n vad python=3.11 -y
conda activate vad
pip install git+https://github.com/TEN-framework/ten-vad.git
pip install pandas soundfile torch torchaudio numpy
```

### Usage

```bash
# Process directory of WAV files (searches recursively)
sbatch scripts/vad.slurm /path/to/audio/directory

# Custom output location
sbatch scripts/vad.slurm /path/to/audio metadata/custom_name
```

**Outputs**:
- `metadata/<dirname>_<date>.csv` - Full VAD statistics (max-spoken, min-spoken, spch-ratio, etc.)
- `metadata/<dirname>_<date>.json` - JSONL manifest with only `audio_filepath` and `duration`

**Use the `.json` file as input for STS and ASR pipelines.**

---

## Pipeline 2: Speech-to-Speech (STS)

**What it does**: Resynthesizes audio through discrete units (mHuBERT encoder + HiFi-GAN vocoder).

**Environment**: Python 3.9, GPU required, uses 2021 fairseq checkpoints

### Setup

```bash
conda create -n sts python=3.9 -y
conda activate sts

# Install textlesslib
git clone https://github.com/facebookresearch/textlesslib.git
pip install -e textlesslib/

# Install specific fairseq commit
pip install git+https://github.com/pytorch/fairseq.git@dd106d9534b22e7db859a6b87ffd7780c38341f8

# Install dependencies with exact versions
pip install 'omegaconf==2.0.6' 'hydra-core==1.0.7' h5py pandas==1.5.3
```

**Critical Fix**: Edit `textlesslib/textless/data/hubert_feature_reader.py` line 32:

```python
# Change this line:
model, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.checkpoint_path])

# To this (add strict=False):
model, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [self.checkpoint_path],
    strict=False,
)
```

This allows loading old checkpoints with incompatible weight shapes. See `TECHNICAL.md` for details.

### Usage

```bash
# Full run with 16 parallel tasks
sbatch --array=0-15 scripts/sts.slurm metadata/input.json dataset_name

# Test with 1 file
sbatch --array=0 scripts/sts.slurm metadata/input.json dataset_name

# Rerun specific failed tasks (maintains original 16-task split)
sbatch --array=0,5,12 scripts/sts.slurm metadata/input.json dataset_name 16
```

**Input**: JSONL manifest with `audio_filepath` and `duration`  
**Output**: `output/<dataset_name>/` - Resynthesized 16kHz WAV files preserving directory structure

**Notes**:
- Automatically resamples input to 16kHz (mHuBERT requirement)
- Skips already-processed files (resume-friendly)
- Dataset name must appear in the input paths (e.g., `/path/to/expresso/audio/file.wav` with `dataset_name=expresso`)

---

## Pipeline 3: Automatic Speech Recognition (ASR)

**What it does**: Transcribes speech using NVIDIA Canary-Qwen-2.5B model.

**Environment**: Python 3.10+, GPU required

### Setup

```bash
conda create -n asr python=3.10 -y
conda activate asr
pip install "nemo_toolkit[all]"
```

### Usage

```bash
# Process with 16 parallel GPU tasks
sbatch --array=0-15 scripts/asr.slurm metadata/input.json

# Single task for testing
sbatch --array=0 scripts/asr.slurm metadata/input.json

# After all tasks complete, merge results
python scripts/merge_manifest.py --manifest metadata/input.json
```

**Input**: JSONL manifest with `audio_filepath` and `duration`  
**Output**: Updates input manifest in-place, adding `"text"` field with transcriptions

**How it works**:
1. Each GPU task reads the full manifest
2. Task N processes files where `index % num_tasks == N` (round-robin)
3. Each task writes task-specific results to `.metadata/<basename>_transcriptions/task_XXXX.json`
4. Merge script combines all task files back into the original manifest
5. Temporary transcription files are cleaned up

**Notes**:
- Auto-detects GPU memory and adjusts batch size (8-16)
- Skips entries that already have a `"text"` field (resume-friendly)
- Uses file locking for safe concurrent writes

---

## Data Format: JSONL Manifests

All pipelines use **newline-delimited JSON** (`.json` extension, JSONL format):

```json
{"audio_filepath": "/absolute/path/to/file1.wav", "duration": 21.4}
{"audio_filepath": "/absolute/path/to/file2.wav", "duration": 33.2}
```

**Important**: Always use **absolute paths** - compute nodes may have different `$PWD`.

After ASR processing, entries have an additional field:
```json
{"audio_filepath": "/absolute/path/file.wav", "duration": 21.4, "text": "transcription here"}
```

---

## File Structure

```
/scratch2/ddager/amela/
├── scripts/
│   ├── vad.py              # VAD processing script
│   ├── vad.slurm           # VAD SLURM launcher
│   ├── sts.py              # STS processing script
│   ├── sts.slurm           # STS SLURM launcher
│   ├── asr.py              # ASR processing script
│   ├── asr.slurm           # ASR SLURM launcher
│   └── merge_manifest.py   # ASR result merger
├── metadata/               # Input/output manifests
├── output/                 # STS resynthesized audio
└── logs/                   # SLURM job logs
```

---

## Common Workflows

### Process new dataset from scratch

```bash
# 1. Generate manifest with VAD
conda activate vad
sbatch scripts/vad.slurm /path/to/dataset/audio/
# Creates: metadata/audio_<date>.json

# 2. Resynthesize audio
conda activate sts
sbatch --array=0-15 scripts/sts.slurm metadata/audio_<date>.json dataset_name
# Creates: output/dataset_name/**/*.wav

# 3. Transcribe audio
conda activate asr
sbatch --array=0-15 scripts/asr.slurm metadata/audio_<date>.json
python scripts/merge_manifest.py --manifest metadata/audio_<date>.json
# Updates: metadata/audio_<date>.json (adds "text" field)
```

### Resume interrupted job

All pipelines skip already-processed files:

```bash
# STS: Skips files that exist in output/
sbatch --array=0-15 scripts/sts.slurm metadata/input.json dataset_name

# ASR: Skips entries with "text" field
sbatch --array=0-15 scripts/asr.slurm metadata/input.json
```

### Check job status

```bash
# List your jobs
squeue -u $USER

# Watch real-time logs
tail -f logs/sts_JOBID_TASKID.out
tail -f logs/asr_JOBID_TASKID.out
tail -f logs/vad_JOBID.out

# Check for errors
grep ERROR logs/*.err
```

---

## Pipeline 4: LSTM Language Model Training

**What it does**: Trains LSTM models on discrete audio tokens for language modeling.

**Environment**: Uses `textless` conda environment (same as STS pipeline)

**Prerequisites**: Audio tokens from encoding pipeline (`.pt` files in `output/librivox_mhubert_expresso_2000/`)

### Setup

Already configured if you've set up the STS pipeline. Uses the same `textless` environment.

### Usage

**Single training run**:
```bash
python scripts/train.py \
    --manifest metadata/librivox_29-10-25.csv \
    --tokens_dir output/librivox_mhubert_expresso_2000 \
    --embedding_dim 256 \
    --hidden_size 512 \
    --num_layers 2 \
    --dropout 0.1 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --num_epochs 100 \
    --early_stopping 10
```

**Grid search with SLURM** (recommended):
```bash
# Submit array job - tests 16 hyperparameter combinations in parallel
sbatch scripts/train.slurm \
    metadata/librivox_29-10-25.csv \
    output/librivox_mhubert_expresso_2000

# Monitor progress
tail -f logs/train_JOBID_TASKID.out
```

**Outputs**:
- `checkpoints/lstm_r{lr}_h{hidden}_e{emb}_l{layers}_b{batch}_d{dropout}/` - Model checkpoints
- `logs/train_JOBID_TASKID.out` - Training logs

**Default Grid Search** (edit `train.slurm` to customize):
- `embedding_dim`: [128, 256]
- `hidden_size`: [256, 512]
- `num_layers`: [2, 3]
- `dropout`: [0.1, 0.2]

Total: 16 combinations tested in parallel

**Key Features**:
- **Nested Tensors**: Efficient variable-length sequences (no padding waste)
- **Memory Optimization**: Loads dataset into RAM (~3GB) for speed
  - Use `--load_on_the_fly` flag if memory-constrained
- **Early Stopping**: Stops after 10 epochs without improvement
- **Automatic Checkpointing**: Saves best model based on validation loss

**Token Format**:
- Vocab: 0-1999 (mHuBERT k-means clusters)
- SOS token: 2000 (prepended to all sequences)
- Total vocabulary: 2001 tokens

---

## Troubleshooting

### STS: "OmegaConf validation error"
- **Cause**: Old checkpoints store integers as floats
- **Fix**: Already handled by monkey-patches in `sts.py`
- **Details**: See `TECHNICAL.md`

### STS: "Weight normalization shape mismatch"
- **Cause**: Old checkpoint format incompatible with modern PyTorch
- **Fix**: Must edit `textlesslib/textless/data/hubert_feature_reader.py` (see Setup above)

### ASR: "Task X wrote but text field missing"
- **Cause**: Task crashed before writing results
- **Fix**: Rerun specific task: `sbatch --array=X scripts/asr.slurm manifest.json`

### VAD: "No WAV files found"
- **Cause**: Directory doesn't contain `.wav` files or wrong path
- **Fix**: Check path and file extensions

### All: "Conda environment not found"
- **Cause**: Environment not activated or doesn't exist
- **Fix**: Follow Setup instructions for each pipeline

---

## SLURM Array Jobs

All GPU pipelines use **array jobs** for parallelization:

```bash
# Run 16 tasks in parallel
sbatch --array=0-15 scripts/sts.slurm ...

# Limit to 7 concurrent tasks (GPU availability)
sbatch --array=0-15%7 scripts/sts.slurm ...

# Run specific tasks only
sbatch --array=0,5,12 scripts/sts.slurm ...
```

**How it works**:
- Each task gets unique `SLURM_ARRAY_TASK_ID` (0, 1, 2, ...)
- Task N processes files where `file_index % num_tasks == N` (round-robin)
- Tasks run independently and can be on different nodes

**Benefits**:
- Near-linear speedup (16 tasks ≈ 16× faster)
- Fault-tolerant (one task fails, others continue)
- Easy to resume (rerun just failed tasks)

---

## Why Three Separate Environments?

**Incompatible dependencies**:

| Pipeline | Python | Key Deps | Reason |
|----------|--------|----------|--------|
| VAD | 3.11+ | TEN-VAD, numpy | Modern libraries, CPU-optimized |
| STS | 3.9 | fairseq@dd106d9, omegaconf==2.0.6 | 2021 checkpoints, old APIs |
| ASR | 3.10+ | NeMo toolkit | GPU-optimized, modern PyTorch |

Mixing these would cause version conflicts and import errors.

---

## Performance Notes

**VAD** (CPU):
- ~100-200 files/sec on 32 cores
- Scales linearly with CPU count
- Bottleneck: I/O for short files

**STS** (GPU):
- ~1 file/sec per GPU (depends on duration)
- Memory: ~2GB per task
- Bottleneck: Encoder inference

**ASR** (GPU):
- ~400× real-time factor per GPU
- Memory: Scales with batch size (8-16)
- Bottleneck: Model forward pass

---

## References

g- [Textlesslib](https://github.com/facebookresearch/textlesslib) - Discrete speech units
- [Fairseq](https://github.com/pytorch/fairseq) - Sequence modeling toolkit
- [TEN-VAD](https://github.com/TEN-framework/ten-vad) - Voice activity detection
- [NeMo](https://github.com/NVIDIA/NeMo) - Speech AI toolkit

For implementation details and compatibility fixes, see **TECHNICAL.md**.
