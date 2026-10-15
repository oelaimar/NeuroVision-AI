import os
import sys

import numpy as np
import tensorflow as tf


sys.path.append('../')
from src.helper import load_dataset

AUTOTUNE = tf.data.AUTOTUNE

def load_dataset_cached(
        raw_dir: str,
        classes: list[str],
        cache_path: str = "data/cache/dataset_224.npz",
        target_size : tuple[int, int] = (224, 224),
        seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data['X'], data['y']
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
        shuffle_buffer: int = 1000,
        seed: int = 42,
        cache_file: str | None = None,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    ds = ds.cache(cache_file) if cache_file else ds.cache()

    if training:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)
    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=AUTOTUNE,
    )
    return ds.prefetch(AUTOTUNE)

def get_datasets(
        raw_dir: str,
        classes: list[str],
        batch_size: int = 32,
        val_split: float = 0.2,
        target_size: tuple[int, int] = (224, 224),
        cache_path: str = "data/cache/dataset_224.npz",
        seed: int = 42,
):
    X, y = load_dataset_cached(raw_dir, classes, cache_path, target_size, seed)
    X_train, y_train, X_val, y_val = stratified_split(X, y, val_split, seed)

    train_ds = make_tf_dataset(X_train, y_train, batch_size, training=True, seed=seed)
    val_ds = make_tf_dataset(X_val, y_val, batch_size, training=False)
    return train_ds, val_ds