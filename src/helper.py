import cv2
import random
import os
from pathlib import Path
import numpy as np

# load images and return a tuple of images and skipped images

def load_image(
        fold_path : str,
        target_size : tuple[int, int] = (224, 224),
) -> tuple[list, int]:
    imgs = []
    count_non_img = 0

    for filename in sorted(os.listdir(fold_path)):
        image_path = os.path.join(fold_path, filename)

        try:
           img = cv2.imread(image_path)
           if img is None:
                raise ValueError("OpenCV could not decode this file")
           img = cv2.resize(img, target_size, interpolation= cv2.INTER_AREA)
           imgs.append(img)
        except Exception as e:
            print(f"WARNING: Could not read {image_path}: {e}")
            count_non_img = count_non_img + 1
    print(f"Successfully loaded: {len(imgs)} images")
    print(f"Skipped: {count_non_img} files")
    return imgs, count_non_img

def load_dataset(
        raw_dir: str,
        classes: list[str],
        target_size: tuple[int, int] = (224, 224),
        shuffle: bool = True,
        seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for label, class_name in enumerate(classes):
        imgs, _ = load_image(os.path.join(raw_dir, class_name), target_size)
        X.extend(imgs)
        y.extend([label] * len(imgs))
    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=np.int64)

    if shuffle:
        idx = np.random.default_rng(seed).permutation(len(X))

        X, y = X[idx], y[idx]
    return X, y