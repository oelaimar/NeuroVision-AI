import os

import numpy as np
import tensorflow as tf

from src.balance import class_distribution, compute_class_weights
from src.helper import load_dataset

AUTOTUNE = tf.data.AUTOTUNE

_augmenter = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),   # ~ +/-18 degrees
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ]
)


def _augment(x, y):
    x = _augmenter(tf.cast(x, tf.float32), training=True)
    return tf.cast(tf.clip_by_value(x, 0, 255), tf.uint8), y


def _normalize(x, y):
    return tf.cast(x, tf.float32) / 255.0, y


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


def stratified_split(X, y, val_split: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    val_idx = []

    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        val_idx.extend(idx[: int(len(idx) * val_split)])

    mask = np.zeros(len(y), dtype=bool)
    mask[val_idx] = True
    return X[~mask], y[~mask], X[mask], y[mask]


def make_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    training: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).map(_normalize, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


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
            extra = (
                ds_c.repeat()
                .map(_augment, num_parallel_calls=AUTOTUNE)
                .take(target - int(n))
            )
            parts.append(extra)

    ds = parts[0]
    for p in parts[1:]:
        ds = ds.concatenate(p)

    ds = ds.shuffle(target * len(counts), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).map(_normalize, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def get_datasets(
    raw_dir: str,
    classes: list[str],
    batch_size: int = 32,
    val_split: float = 0.2,
    target_size: tuple[int, int] = (224, 224),
    cache_path: str = "data/cache/dataset_224.npz",
    seed: int = 42,
    strategy: str = "auto",  # "auto" | "none" | "class_weight" | "augment"
):
    X, y = load_dataset_cached(raw_dir, classes, cache_path, target_size, seed)
    X_train, y_train, X_val, y_val = stratified_split(X, y, val_split, seed)

    counts = class_distribution(y_train, classes, "(train)")
    class_distribution(y_val, classes, "(val)")

    if strategy == "auto":
        ratio = counts.max() / counts.min()
        strategy = "none" if ratio < 1.5 else "class_weight" if ratio < 3 else "augment"
    print(f"\nRebalancing strategy: {strategy}")

    class_weight = None
    if strategy == "augment":
        train_ds = make_balanced_train_ds(X_train, y_train, batch_size, seed)
    else:
        train_ds = make_tf_dataset(X_train, y_train, batch_size, training=True, seed=seed)
        if strategy == "class_weight":
            class_weight = compute_class_weights(y_train, len(classes))
            print("class_weight:", {classes[k]: round(v, 3) for k, v in class_weight.items()})

    val_ds = make_tf_dataset(X_val, y_val, batch_size, training=False)
    return train_ds, val_ds, class_weight