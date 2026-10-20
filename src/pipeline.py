import hashlib
import os
import json
import numpy as np
import tensorflow as tf
from dataclasses import dataclass, asdict
from functools import lru_cache

from src.balance import class_distribution, compute_class_weights
from src.helper import load_dataset

AUTOTUNE = tf.data.AUTOTUNE
_test_used = False

# Single definition of "the data"

@dataclass(frozen=True)
class DataSpec:
    raw_dir: str
    classes: tuple[str, ...]
    val_split: float = 0.15
    test_split: float = 0.15
    target_size: tuple[int, int] = (224, 224)
    cache_path: str = "data/cache/dataset_224.npz"
    manifest_path: str = "data/splits_manifest.json"
    seed: int = 42
    dedup: bool = True

    def key(self) -> str:
        """Hash of everything that changes the data or the split."""
        d = asdict(self)
        d.pop("raw_dir")
        d.pop("cache_path")
        d.pop("manifest_path")
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=list).encode()).hexdigest()[:8]

    def keyed(self, path: str) -> str:
        base, ext = os.path.splitext(path)
        return f"{base}_{self.key()}{ext}"


def build_augmenter(seed: int = 42) -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.03, fill_mode="constant", fill_value=0.0, seed=seed),
            tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="constant", fill_value=0.0, seed=seed),
            tf.keras.layers.RandomZoom(0.1, fill_mode="constant", fill_value=0.0, seed=seed),
            tf.keras.layers.RandomBrightness(0.1, value_range=(0, 255), seed=seed),
            tf.keras.layers.RandomContrast(0.1, seed=seed),
        ],
        name="mri_augmentation",
    )

def _finalize(ds: tf.data.Dataset, batch_size: int, augment:bool, seed: int) -> tf.data.Dataset:
    opts = tf.data.Options()
    opts.deterministic = True
    ds = ds.with_options(opts).batch(batch_size)

    augmenter = build_augmenter(seed) if augment else None

    def prep(x, y):
        x = tf.cast(x, tf.float32)
        if augmenter is not None:
            x = tf.clip_by_value(augmenter(x, training=True), 0.0, 255.0)
        return x / 255.0, y

    ds = ds.map(prep, num_parallel_calls=1 if augment else AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

def _load_cached(spec: DataSpec) -> tuple[np.ndarray, np.ndarray]:
    path = spec.keyed(spec.cache_path)
    if os.path.exists(path):
        with np.load(path) as data:
            if list(data["classes"]) != list(spec.classes):
                raise ValueError(
                    f"Cache {path} has a different class order. Delete it."
                )
            X = data["X"].copy()
            y = data["y"].astype(np.int64)
        return X, y
    X, y = load_dataset(
        spec.raw_dir,
        list(spec.classes),
        target_size=spec.target_size,
        seed=spec.seed,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, X=X, y=y, classes=np.array(spec.classes))
    return X, y.astype(np.int64)

def remove_duplicates(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop exact duplicate images BEFORE splitting, so the same image can't
        land in both train and val/test."""
    seen, keep = set(), []
    for i, img in enumerate(X):
        h = hashlib.md5(img.tobytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            keep.append(i)
    keep = np.array(keep)
    print(f"Duplicates removed: {len(X) - len(keep)} (kept {len(keep)})")
    return X[keep], y[keep]

def stratified_split_indices(y, val_split: float, test_split: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = int(len(idx) * test_split)
        n_val = int(len(idx) * val_split)
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test:n_test + n_val])
        train_idx.extend(idx[n_test + n_val:])

    train_idx, val_idx, test_idx = map(np.array, (train_idx, val_idx, test_idx))
    assert not (set(train_idx) & set(val_idx) | set(train_idx) & set(test_idx) | set(val_idx) & set(test_idx)), \
        "Leakage: splits overlap"
    return train_idx, val_idx, test_idx

def _save_or_verify_manifest(spec: DataSpec, idx) -> None:
    """Save split indices once; on later runs verify they are identical."""
    path = spec.keyed(spec.manifest_path)
    tr, va, te = idx
    digest = hashlib.sha256(json.dumps([tr.tolist(), va.tolist(), te.tolist()]).encode()).hexdigest()[:12]
    if os.path.exists(path):
        with open(path) as f:
            if json.load(f)["sha256_12"] != digest:
                raise RuntimeError(f"Split drift vs {path}: data or load order changed. Delete it to regenerate.")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "spec": asdict(spec),
            "sha256_12": digest,
            "n": {"train": len(tr), "val": len(va), "test": len(te)},
            "train_idx": tr.tolist(), "val_idx": va.tolist(), "test_idx": te.tolist(),
        }, f)

@lru_cache
def get_splits(spec: DataSpec):
    """Load -> dedup -> stratified split. Cached: train, val and test share one run."""
    X, y = _load_cached(spec)
    if spec.dedup:
        X, y = remove_duplicates(X, y)

    idx = stratified_split_indices(y, spec.val_split, spec.test_split, spec.seed)
    _save_or_verify_manifest(spec, idx)
    tr, va, te = idx
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def make_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    training: bool = False,
    augment: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    return _finalize(ds, batch_size, augment=training and augment, seed=seed)


def make_balanced_train_ds(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    seed: int = 42,
) -> tf.data.Dataset:
    counts = np.bincount(y)
    target = int(counts.max())

    parts = []
    for c, n in enumerate(counts):
        ds_c = tf.data.Dataset.from_tensor_slices((X[y == c], y[y == c]))
        parts.append(ds_c)
        if n < target:
            parts.append(ds_c.repeat().take(target - int(n)))

    ds = parts[0]
    for p in parts[1:]:
        ds = ds.concatenate(p)

    ds = ds.shuffle(target * len(counts), seed=seed, reshuffle_each_iteration=True)
    return _finalize(ds, batch_size, augment=True, seed=seed)


def get_datasets(
        spec: DataSpec,
        batch_size: int = 32,
        strategy: str = "auto",  # "auto" | "none" | "class_weight" | "augment"
        augment: bool = True
):
    classes, seed = list(spec.classes), spec.seed
    (X_train, y_train), (X_val, y_val), (_, y_test) = get_splits(spec)

    counts = class_distribution(y_train, classes, "(train)")
    class_distribution(y_val, classes, "(val)")
    class_distribution(y_test, classes, "(test, counts only)")

    if strategy == "auto":
        ratio = counts.max() / counts.min()
        strategy = "none" if ratio < 1.5 else "class_weight" if ratio < 3 else "augment"
    print(f"\nRebalancing strategy: {strategy} | augmentation: {augment or strategy == 'augment'}")

    class_weight = None
    if strategy == "augment":
        train_ds = make_balanced_train_ds(X_train, y_train, batch_size, seed)
    else:
        train_ds = make_tf_dataset(X_train, y_train, batch_size, training=True, augment=augment, seed=seed)
        if strategy == "class_weight":
            class_weight = compute_class_weights(y_train, len(classes))
            print("class_weight:", {classes[k]: round(v, 3) for k, v in class_weight.items()})

    val_ds = make_tf_dataset(X_val, y_val, batch_size, training=False)
    return train_ds, val_ds, class_weight

def get_test_ds(spec: DataSpec, batch_size: int = 32, allow_reuse: bool = False) -> tf.data.Dataset:
    global _test_used
    if _test_used and not allow_reuse:
        raise RuntimeError(
            "Test set already used. Reusing it for decisions leaks it into your "
            "model selection. Pass allow_reuse=True only if you accept that."
        )
    _, _, (X_test, y_test) = get_splits(spec)
    _test_used = True
    return make_tf_dataset(X_test, y_test, batch_size, training=False)