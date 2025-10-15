# AMELA - Acoustic Modeling for Early Language Acquisition

This repository contains scripts for speech-to-speech resynthesis using textlesslib with fairseq models.

## Installation S2S Pipeline

```bash
# Create conda environment
conda create -n textless python=3.9 -y
conda activate textless

# Clone and install textlesslib
git clone git@github.com:facebookresearch/textlesslib.git
pip install -e textlesslib/

# Install specific fairseq commit (required for checkpoint compatibility)
pip install git+https://github.com/pytorch/fairseq.git@dd106d9534b22e7db859a6b87ffd7780c38341f8

# Install compatible OmegaConf and Hydra versions
pip install 'omegaconf==2.0.6' 'hydra-core==1.0.7' 'librosa==0.9.2'

# Verify installation
python -c "import fairseq; import omegaconf; import hydra; import librosa; print('fairseq:', fairseq.__version__); print('omegaconf:', omegaconf.__version__); print('hydra:', hydra.__version__); print('librosa:', librosa.__version__)"
```

Expected output:

```
fairseq: 1.0.0a0+dd106d9
omegaconf: 2.0.6
hydra: 1.0.7
librosa: 0.9.2
```

### Critical: Apply Compatibility Fix

After installation, modify one line in textlesslib:

**File**: `textlesslib/textless/data/hubert_feature_reader.py`  
**Line 32-34**: Add `strict=False` parameter

```python
# Change this:
model, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [self.checkpoint_path]
)

# To this:
model, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [self.checkpoint_path],
    strict=False,  # Allow loading old checkpoints with incompatible weights
)
```

## Usage

```bash
conda activate textless
srun --partition=gpu --gres=gpu:1 --mem=32G python scripts/speech2speech.py
```

## Why These Specific Versions?

Old fairseq checkpoints (2021) have compatibility issues with modern libraries:

- **OmegaConf 2.0.x**: Has `II` interpolation (needed by fairseq) but not overly strict validation
- **Hydra 1.0.x**: Compatible with OmegaConf 2.0.x
- **Fairseq dd106d9**: The exact commit textlesslib was developed against

**For detailed technical explanation, see `COMPATIBILITY.md`**

## Troubleshooting

| Error                                           | Solution                                         |
| ----------------------------------------------- | ------------------------------------------------ |
| `Value 'X.0' could not be converted to Integer` | `pip install 'omegaconf>=2.0,<2.1'`              |
| `cannot import name 'II' from 'omegaconf'`      | `pip install 'omegaconf>=2.0,<2.1'`              |
| `size mismatch for encoder.pos_conv.0.weight_g` | Add `strict=False` in `hubert_feature_reader.py` |
| `No config source registered for schema pkg`    | `pip install 'hydra-core>=1.0,<1.1'`             |

## References

- [Textlesslib](https://github.com/facebookresearch/textlesslib)
- [Fairseq](https://github.com/pytorch/fairseq)

## Installation VAD Pipeline

```bash
# Note that if you did not install libc++1, you have to run the code below to install it:
sudo apt update
sudo apt install libc++1

conda create -n vad python=3.9 -y
conda activate vad

pip install git+https://github.com/TEN-framework/ten-vad.git@aa96832d58a295d97b9a6baa4109a9bede4474f8
```
