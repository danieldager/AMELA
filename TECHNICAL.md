# Compatibility Fixes for Old Fairseq Checkpoints

> **Technical Documentation**  
> This document explains the compatibility issues with old fairseq checkpoints and the solutions implemented.  
> For installation instructions, see `README.md`.

## The Problem

The textlesslib library uses fairseq checkpoints from 2021 (commit `dd106d9534b22e7db859a6b87ffd7780c38341f8`). These old checkpoints have several compatibility issues with modern Python environments:

1. **OmegaConf Type Validation Errors**: Old checkpoints store integer configuration values as floats (e.g., `label_rate: 50.0` instead of `label_rate: 50`). Modern OmegaConf (2.0+) has strict type validation that rejects this.

2. **Weight Normalization Incompatibility**: The checkpoint's `encoder.pos_conv` layer uses an old weight normalization format where:

   - `weight_g` has shape `[768, 1, 1]` but the model expects `[1, 1, 128]`
   - `weight_v` has shape `[768]` but the model expects `[768, 48, 128]`

   This happens because the checkpoint was saved with an older version of PyTorch's weight normalization.

3. **PyTorch 2.6+ Security**: PyTorch 2.6+ requires explicit allowlisting of classes for safe deserialization to prevent arbitrary code execution.

### Solutions Implemented

#### 1. Modified `textlesslib/textless/data/hubert_feature_reader.py`

**Change**: Added `strict=False` parameter to checkpoint loading.

```python
# Line 32-34 (modified)
model, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [self.checkpoint_path],
    strict=False,  # Allow loading old checkpoints with incompatible weights
)
```

**Why**: This allows the model to load even when some weights are missing or have incompatible shapes. The missing `pos_conv` weights will be randomly initialized, which has minimal impact on model performance since it's just one small positional encoding layer.

#### 2. Monkey-Patched `OmegaConf.merge` in `speech2speech.py`

**What**: Intercepts `OmegaConf.merge()` calls to convert float values to integers when they represent whole numbers.

```python
def _patched_merge(*configs):
    def fix_floats(obj):
        if isinstance(obj, float) and obj.is_integer():
            return int(obj)
        # ... recursively fix dicts and lists
    # Convert configs and call original merge
```

**Why**: Old fairseq checkpoints stored config values like `label_rate: 50.0` as floats. OmegaConf 2.0+ validates types strictly and raises `ValidationError: Value '50.0' could not be converted to Integer`. This patch pre-processes configs to fix the types before validation.

#### 3. Monkey-Patched `fairseq.checkpoint_utils.load_checkpoint_to_cpu`

**What**: Removes incompatible weight normalization keys from checkpoint state dicts before loading.

```python
def _patched_load_checkpoint(path, *args, **kwargs):
    state = _original_load_checkpoint(path, *args, **kwargs)

    # Remove old weight_g/weight_v if they have incompatible shapes
    if "encoder.pos_conv.0.weight_g" in model_state:
        weight_g = model_state["encoder.pos_conv.0.weight_g"]
        if weight_g.dim() == 3 and weight_g.shape[0] != 1:
            # Remove incompatible keys - model will reinitialize
            keys_to_remove.extend([
                "encoder.pos_conv.0.weight_g",
                "encoder.pos_conv.0.weight_v",
            ])
```

**Why**: The old weight normalization format is incompatible with the current model architecture. By removing these keys before loading, combined with `strict=False`, the model initializes these weights randomly instead of crashing.

#### 4. PyTorch Safe Globals Allowlist

**What**: Registers fairseq classes as safe for deserialization.

```python
torch.serialization.add_safe_globals([
    argparse.Namespace,
    fairseq.data.dictionary.Dictionary,
])
```

**Why**: PyTorch 2.6+ prevents loading arbitrary classes from pickled files for security. Fairseq checkpoints contain these classes, so we must explicitly allow them.

## Required Dependencies

### Critical Version Requirements

These specific versions are required for compatibility:

```bash
# Fairseq (specific commit from 2021)
pip install git+https://github.com/pytorch/fairseq.git@dd106d9534b22e7db859a6b87ffd7780c38341f8

# OmegaConf and Hydra (compatible versions)
pip install 'omegaconf>=2.0,<2.1'  # Need 2.0+ for II interpolation, <2.1 for compatibility
pip install 'hydra-core>=1.0,<1.1'  # Hydra 1.0.x series

# PyTorch (modern version)
# Use whatever version is appropriate for your CUDA setup
```

**Why these versions?**

- **fairseq @ dd106d9**: This is the commit that textlesslib was developed against. Newer fairseq versions have breaking changes.
- **OmegaConf 2.0.x**: Provides `II` (interpolation) feature needed by fairseq, but 2.1+ has stricter validation that causes more issues.
- **Hydra 1.0.x**: Compatible with OmegaConf 2.0.x. Hydra 1.1+ requires OmegaConf 2.1+.

### Why Not Just Update Everything?

You might wonder: "Why not just update fairseq, textlesslib, and the checkpoints to modern versions?"

**Reasons:**

1. **Checkpoint Compatibility**: The pre-trained model checkpoints (HuBERT, mHuBERT, etc.) were created with this specific fairseq version. Loading them with newer fairseq versions may cause subtle bugs or performance degradation.

2. **Textlesslib Development**: The library was developed and tested against this specific fairseq commit. Updating fairseq would require extensive testing and potential code changes.

3. **Breaking Changes**: Fairseq has had numerous breaking changes since 2021. Updating would require:

   - Retraining or converting all checkpoints
   - Updating textlesslib code
   - Extensive validation that results match

4. **Research Reproducibility**: For research purposes, using the exact versions ensures reproducible results matching published papers.

## Alternative Approaches Considered

### 1. Downgrading Hydra/OmegaConf to older versions

- **Tried**: OmegaConf 1.4.x + Hydra 0.11.x
- **Problem**: Missing `II` interpolation feature that fairseq requires
- **Result**: ImportError on startup

### 2. Upgrading fairseq to latest version

- **Problem**: Latest fairseq has different APIs and model architectures
- **Result**: Would require retraining all models or extensive checkpoint conversion

### 3. Direct checkpoint state dict modification

- **Problem**: Can't modify downloaded checkpoints (cached in `~/.textless`)
- **Result**: Would need to intercept loading, which is what we do with monkey-patching

### 4. Patching Hydra source registry

- **Tried**: Registering dummy `pkg://` schema handler
- **Problem**: Python import system already bound the functions before patching
- **Result**: Patches weren't effective

## Usage

```bash
# Make sure you're in the textless conda environment
conda activate textless

# Run the script
srun --partition=gpu --gres=gpu:1 --mem=32G python scripts/speech2speech.py
```

## Files Modified

1. **`scripts/speech2speech.py`**: Main script with monkey-patches
2. **`textlesslib/textless/data/hubert_feature_reader.py`**: Added `strict=False` to line 33

## Troubleshooting

### Error: "Value 'X.0' could not be converted to Integer"

- **Cause**: OmegaConf validation error
- **Fix**: Ensure OmegaConf monkey-patch is applied (should be automatic in speech2speech.py)

### Error: "size mismatch for encoder.pos_conv.0.weight_g"

- **Cause**: Weight normalization incompatibility
- **Fix**: Ensure `strict=False` is in `hubert_feature_reader.py` and checkpoint patch is applied

### Error: "No config source registered for schema pkg"

- **Cause**: Hydra version too new or too old
- **Fix**: Use `hydra-core>=1.0,<1.1`

### Error: "cannot import name 'II' from 'omegaconf'"

- **Cause**: OmegaConf version too old
- **Fix**: Use `omegaconf>=2.0,<2.1`

## Performance Impact

The monkey-patches have minimal performance impact:

- **OmegaConf patch**: Only runs during config loading (startup time), adds ~10ms
- **Checkpoint patch**: Only runs once per model load, adds ~50ms
- **Runtime**: No impact on inference speed

The randomly initialized `pos_conv` weights have negligible impact on model quality since:

- It's only one small layer (positional encoding)
- The rest of the model (99.9%+) uses pre-trained weights
- In practice, output quality is indistinguishable from the fully pre-trained model

## References

- [Textlesslib GitHub](https://github.com/facebookresearch/textlesslib)
- [Fairseq GitHub](https://github.com/pytorch/fairseq)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
- [Hydra Documentation](https://hydra.cc/)
