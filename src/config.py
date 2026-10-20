from dataclasses import dataclass, asdict
from typing import Tuple

@dataclass(frozen=True)
class Config:
    # --- reproducibility ---
    seed: int = 42
    deterministic_ops: bool = True

    # --- data ---
    data_dir: str = "data/raw"
    classes: Tuple[str, ...] = ("glioma", "meningioma", "notumor", "pituitary")
    img_size: Tuple[int, int] = (224, 224)
    val_frac: float = 0.15
    test_frac: float = 0.15
    cache_path: str = "data/cache/dataset_224.npz"
    manifest_path: str = "data/splits_manifest.json"

    # --- training ---
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    strategy: str = "auto"  # "auto" | "none" | "class_weight" | "augment"
    dropout: float = 0.2
    use_class_weight: bool = True

    # --- MobileNetV2 ---

    head_epochs: int = 10
    finetune_epochs: int = 15
    finetune_lr: float = 1e-5
    finetune_layers: int = 30

    # --- outputs ---
    runs_dir: str  = "runs"

    def to_dict(self) -> dict:
        return asdict(self)