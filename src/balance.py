import numpy as np


def class_distribution(y: np.ndarray, classes: list[str], name: str = "") -> np.ndarray:
    counts = np.bincount(y, minlength=len(classes))
    total = counts.sum()

    print(f"\nClass distribution {name}".rstrip())
    for c, n in zip(classes, counts):
        bar = "#" * int(40 * n / counts.max())
        print(f"  {c:<12} {n:>6}  {100 * n / total:5.1f}%  {bar}")

    ratio = counts.max() / counts.min()
    if ratio < 1.5:
        verdict = "balanced enough: no rebalancing needed"
    elif ratio < 3:
        verdict = "moderate imbalance: use class_weight"
    else:
        verdict = "severe imbalance: use targeted augmentation (+ check per-class recall)"
    print(f"  imbalance ratio (max/min) = {ratio:.2f} -> {verdict}")
    return counts


def compute_class_weights(y: np.ndarray, num_classes: int) -> dict[int, float]:
    """weight_c = N / (num_classes * count_c): rare classes get weight > 1."""
    counts = np.bincount(y, minlength=num_classes)
    weights = len(y) / (num_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}