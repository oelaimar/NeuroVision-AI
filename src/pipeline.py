import hashlib
import os

import numpy as np
import tensorflow as tf

from src.balance import class_distribution, compute_class_weights
from src.helper import load_dataset

AUTOTUNE = tf.data.AUTOTUNE
_test_used = False

def build_augmenter() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.03, fill_mode="constant", fill_value=0.0),
            tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="constant", fill_value=0.0),
            tf.keras.layers.RandomZoom(0.1, fill_mode="constant", fill_value=0.0),
            tf.keras.layers.RandomBrightness(0.1, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.GaussianNoise(5.0),
        ],
        name="mri_augmentation",
    )

_augmenter = build_augmenter()

def _normalize(x, y):
    return tf.cast(x, tf.float32) / 255.0, y

def _augment_and_normalize(x, y):
    x = _augmenter(tf.cast(x, tf.float32), training=True)
    x = tf.clip_by_value(x, 0.0, 255.0)
    return x / 255.0, y

def _finalize(ds: tf.data.Dataset, batch_size: int, augment:bool) -> tf.data.Dataset:
    ds = ds.batch(batch_size)
    ds = ds.map(_augment_and_normalize if augment else _normalize, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

def load_dataset_cached(
    raw_dir: str,
    classes: list[str],
    cache_path: str = "data/cache/dataset_224.npz",
    target_size: tuple[int, int] = (224, 224),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["X"], data["y"]

    X, y = load_dataset(raw_dir, classes, target_size=target_size, seed=seed)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, X=X, y=y)
    return X, y

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

def stratified_split(X, y, val_split: float = 0.15, test_split: float = 0.15, seed: int = 42):
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
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx])

def get_splits(
        raw_dir:str,
        classes: list[str],
        val_split: float = 0.15,
        test_split: float = 0.15,
        target_size: tuple[int, int] = (224, 224),
        cache_path: str = "data/cache/dataset_224.npz",
        seed: int = 42,
        dedup: bool = True,
):
    """Load -> dedup -> stratified split. Nothing is augmented or fitted yet."""
    X, y = load_dataset_cached(raw_dir, classes, cache_path, target_size, seed)
    if dedup:
        X, y = remove_duplicates(X, y)
    return stratified_split(X, y, val_split, test_split, seed)


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
    return _finalize(ds, batch_size, augment=training and augment)


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
    return _finalize(ds, batch_size, augment=True)


def get_datasets(
    raw_dir: str,
    classes: list[str],
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    target_size: tuple[int, int] = (224, 224),
    cache_path: str = "data/cache/dataset_224.npz",
    seed: int = 42,
    strategy: str = "auto",  # "auto" | "none" | "class_weight" | "augment"
    augment: bool = True,
    dedup: bool = True,
):
    (X_train, y_train), (X_val, y_val), (_, y_test) = get_splits(
        raw_dir, classes, val_split, test_split, target_size, cache_path, seed, dedup
    )

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

def get_test_ds(
    raw_dir: str,
    classes: list[str],
    batch_size: int = 32,
    val_split: float = 0.15,      # must match get_datasets
    test_split: float = 0.15,     # must match get_datasets
    target_size: tuple[int, int] = (224, 224),
    cache_path: str = "data/cache/dataset_224.npz",
    seed: int = 42,               # must match get_datasets
    dedup: bool = True,           # must match get_datasets
    allow_reuse: bool = False,
) -> tf.data.Dataset:
    global _test_used
    if _test_used and not allow_reuse:
        raise RuntimeError(
            "Test set already used. Reusing it for decisions leaks it into your "
            "model selection. Pass allow_reuse=True only if you accept that."
        )
    _, _, (X_test, y_test) = get_splits(
        raw_dir, classes, val_split, test_split, target_size, cache_path, seed, dedup
    )
    _test_used = True
    return make_tf_dataset(X_test, y_test, batch_size, training=False)