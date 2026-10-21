# Reproducibility

## What is controlled
- Seeds: `Config.seed` drives python, numpy, TensorFlow (`set_random_seed`), the train/val/test split, `tf.data` shuffle and augmentation seeds.
- Split: generated once with a stratified split on a sorted file listing, saved to `data/splits.json` (relative paths + labels + class order). Reused on every run. Delete the file to regenerate.
- Class order: `Config.classes` (index = label). Checked against the saved splits on load.
- Config: all values in `src/config.py`; each run writes `runs/<ts>_<name>_seed<N>/config.json`, `env.json`, `splits_ref.json`.

## Rerun a result
1. `git checkout <commit from env.json>`
2. Same `data/splits.json` (check hash in `splits_ref.json`)
3. Same environment (`logs/env_versions.txt`, same GPU model if possible)
4. `python -m src.train`

## Known sources of nondeterminism
- GPU model/driver/cuDNN/CUDA versions: results can differ across different GPU types, even with fixed seeds.
- Non-deterministic CUDA kernels (atomics in some conv/pooling backward ops): mitigated by `enable_op_determinism()` at some speed cost; ops with no deterministic version raise an error.
- `PYTHONHASHSEED`: only effective if set before Python starts. Run with `PYTHONHASHSEED=42 python -m src.train` for strict control.
- `tf.data` parallelism: `deterministic=True` is required, otherwise parallel map/interleave can reorder elements.
- Augmentation randomness: only reproducible if every random op has an explicit seed.
- OpenCV/threading: image decode is deterministic, but resize interpolation can vary slightly across OpenCV builds. Pin the version.
- Mixed precision / XLA: can change numerics. Keep settings in config if enabled.
- Dataset changes: adding/removing files changes the split. Keep `splits.json` under version control (or record its hash).
- Library upgrades: pinned in `../scripts/requirements-gpu.txt`; actual versions in `logs/env_versions.txt`.