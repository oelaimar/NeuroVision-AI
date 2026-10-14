
import os
import sys

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src.helper import load_dataset

RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def test_load_dataset():
    X, y = load_dataset(RAW_DIR, CLASSES)

    assert X.shape[1:] == (224, 224, 3), f"Wrong image shape: {X.shape}"
    assert len(X) == len(y), "Images and labels count differ"
    assert set(np.unique(y)) == set(range(len(CLASSES))), "Missing a class"
    assert not np.array_equal(y, np.sort(y)), "Labels are not shuffled"

    for label, class_name in enumerate(CLASSES):
        print(f"{class_name}: {(y == label).sum()} images")

    print(f"\nTotal: {len(X)} images, shape {X.shape}")
    print("All checks passed!")


if __name__ == "__main__":
    test_load_dataset()