import json
import hashlib
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config import Config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def _list_files(cfg: Config):
    """Sorted, relative, deterministic listing. Labels come from cfg.classes order."""
    root = Path(cfg.data_dir)
    paths, labels = [], []
    for label, cls in enumerate(cfg.classes):
        files = sorted(
            p.relative_to(root).as_posix()
            for p in (root / cls).rglob("*")
            if p.suffix.lower() in IMG_EXT
        )
        if not files:
            raise FileExistsError(f"No images found for class '{cls}' in {root / cls}")
        paths += files
        labels += [label] * len(files)
    return paths, labels

def make_splits(cfg: Config) -> dict:
    paths, labels = _list_files(cfg)
    holdout = cfg.val_frac + cfg.test_frac

    tr_p, tmp_p, tr_y, tmp_y = train_test_split(
        paths, labels, test_size=holdout, stratify=labels, random_state=cfg.seed
    )
    val_share = cfg.val_frac / holdout
    va_p, te_p, va_y, te_y = train_test_split(
        tmp_p, tmp_y, test_size=1 - val_share, stratify=tmp_y, random_state=cfg.seed
    )

    return {
        "seed": cfg.seed,
        "classes": list(cfg.classes),
        "val_frac": cfg.val_frac,
        "test_frac": cfg.test_frac,
        "train": {"paths": tr_p, "labels": tr_y},
        "val": {"paths": va_p, "labels": va_y},
        "test": {"paths": te_p, "labels": te_y},
    }

def splits_hash(splits: dict) -> str:
    return hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()[:12]


def load_or_create_splits(cfg: Config) -> dict:
    """Reuse the saved split if present (and consistent), otherwise generate and save."""
    p = Path(cfg.split_path)
    if p.exists():
        splits = json.loads(p.read_text())
        if splits["classes"] != list(cfg.classes):
            raise ValueError("Saved splits use a different class order than config. Delete or regenerate.")
        return splits
    splits = make_splits(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(splits, indent=2))
    return splits