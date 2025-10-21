# Technical Documentation

Implementation details, compatibility fixes, and advanced patterns for AMELA pipelines.

## Speech-to-Speech: Legacy Fairseq Compatibility

The S2S pipeline uses 2021 fairseq checkpoints with modern PyTorch. Three compatibility issues required fixes:

### Issue 1: OmegaConf Type Validation
Old checkpoints store integers as floats (`label_rate: 50.0`). OmegaConf 2.0+ validates strictly and rejects this.

**Solution**: Monkey-patch `OmegaConf.merge()` to convert floats to ints before validation (see `scripts/sts.py` lines 1-50).

### Issue 2: Weight Normalization Shape Mismatch
Checkpoint's `encoder.pos_conv` weights have incompatible shapes:
- `weight_g`: `[768, 1, 1]` but model expects `[1, 1, 128]`
- `weight_v`: `[768]` but model expects `[768, 48, 128]`

**Solution**: Monkey-patch checkpoint loading to remove incompatible keys before loading. Combined with `strict=False`, model reinitializes these weights (~0.1% of parameters, negligible impact).

### Issue 3: PyTorch 2.6+ Security
PyTorch now requires explicit allowlisting of classes for deserialization.

**Solution**: Register fairseq classes with `torch.serialization.add_safe_globals()`.

### Implementation Details

**Modified file**: `textlesslib/textless/data/hubert_feature_reader.py` line 33
```python
model, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [self.checkpoint_path], strict=False  # ADD THIS
)
```

**Monkey patches** in `scripts/sts.py`:
1. Patch `OmegaConf.merge()` to convert floats → ints
2. Patch `load_checkpoint_to_cpu()` to remove incompatible weight keys
3. Register safe classes with `torch.serialization.add_safe_globals()`

**Critical**: Patches must run BEFORE importing fairseq/textless.

### Required Versions
```bash
fairseq @ git+...@dd106d9  # Exact commit textlesslib uses
omegaconf==2.0.6           # Has II interpolation, not overly strict
hydra-core==1.0.7          # Compatible with OmegaConf 2.0.x
```

Why not update? Pre-trained checkpoints are tied to this fairseq version. Updating would require retraining models or extensive checkpoint conversion. For research reproducibility, we use exact versions.

### Performance Impact
- Monkey patches: +10ms startup, +50ms model load, 0ms runtime
- Reinitialized `pos_conv` weights: ~0.1% of parameters, negligible quality impact

## VAD: Multiprocessing Patterns

VAD uses TEN-VAD which cannot be pickled. **Solution**: Initialize model per-process.

```python
# WRONG: Model instance can't be passed through executor
model = TenVad()
with ProcessPoolExecutor() as executor:
    executor.map(process_func, files, repeat(model))  # FAILS

# CORRECT: Initialize per-process
def process_func(args):
    model = TenVad()  # Each worker gets own instance
    return model.process(args)

with ProcessPoolExecutor() as executor:
    executor.map(process_func, files)  # WORKS
```

See `scripts/vad.py` function `process_wavs_optimized()` for full implementation.

## ASR: GPU Memory Auto-Tuning

The SLURM script detects GPU memory and adjusts batch size:

```bash
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
if [ "$GPU_MEM" -gt 70000 ]; then BATCH_SIZE=16; fi    # A100 80GB
elif [ "$GPU_MEM" -gt 35000 ]; then BATCH_SIZE=8; fi   # A100 40GB
else BATCH_SIZE=4; fi                                   # V100 32GB
```

These are conservative. Actual memory usage depends on audio length and max tokens. Monitor with `nvidia-smi` and adjust.

## ASR: Parallel Processing with Job Arrays

For large datasets, use embarrassingly parallel approach (not DDP):

1. **Split manifest** into N splits
2. **Submit job array** with N tasks, each gets 1 GPU
3. **Merge results** after completion

This is simpler than DDP for inference, fault-tolerant (rerun individual failed tasks), and scales to 100+ GPUs.

```bash
python scripts/split_manifest.py --input data.json --splits 30
export SPLITS_DIR="splits"
sbatch --array=0-29 scripts/asr.slurm
python scripts/merge_results.py --input-pattern "output/*_split_*.json" --output final.json
```

See scripts for implementation details. For future **training** workloads (LSTM/Transformer grid search), you'll need `torch.distributed` with DDP.

## HPC Best Practices

### SLURM Script Pattern
1. **Validation** - Check inputs exist before expensive operations
2. **Environment** - Install deps in job (dependencies change, not pre-built)
3. **Auto-optimization** - Detect resources, adjust parameters
4. **Execution** - Run with logging

### Data Manifests
- Always use **absolute paths** (compute nodes have different $PWD)
- Use **JSONL format** (one JSON per line, no top-level array)
- Include **metadata** (duration, etc.) for progress estimation

### Multiprocessing
- For CPU-bound: Use `ProcessPoolExecutor`, workers = CPU count
- For GPU-bound: Use job arrays, 1 GPU per task
- For non-picklable objects: Initialize per-process/per-task

## Common Errors & Solutions

| Error | Cause | Fix |
|-------|-------|-----|
| `Value 'X.0' could not be converted to Integer` | OmegaConf validation | Check monkey-patch in sts.py |
| `size mismatch for encoder.pos_conv` | Weight shape incompatibility | Add `strict=False` in hubert_feature_reader.py |
| `cannot import name 'II'` | OmegaConf too old | `pip install 'omegaconf==2.0.6'` |
| `No config source for pkg` | Hydra version | `pip install 'hydra-core==1.0.7'` |
| ASR CUDA OOM | Batch size too large | Reduce in asr.slurm |
| VAD hangs | Model not picklable | Initialize per-process |
| Empty transcriptions | Audio >40s | Split files or filter manifest |

## References

- [Textlesslib](https://github.com/facebookresearch/textlesslib)
- [Fairseq](https://github.com/pytorch/fairseq)
- [TEN-VAD](https://github.com/TEN-framework/ten-vad)
- [NeMo](https://github.com/NVIDIA/NeMo)
