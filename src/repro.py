import json
import os
import platform
import random
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import Config

def set_seed(cfg: Config) -> None:
    """Seed python, numpy and TensorFlow; optionally force deterministic GPU ops."""
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    if cfg.deterministic_ops:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    import tensorflow as tf

    tf.keras.utils.set_random_seed(cfg.seed)
    if cfg.deterministic_ops:
        tf.config.experimental.enable_op_determinism()

def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unavailable"

def env_info() -> dict:
    import tensorflow as tf
    import cv2
    build = tf.sysconfig.get_build_info()
    return{
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "tf_cuda": build.get("cuda_version"),
        "tf_cudnn": build.get("cudnn_version"),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "gpus": _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": _run(["git", "status", "--porcelain"]) != "",
    }

def start_run(cfg: Config, spec, name: str = "run") -> Path:
    manifest = Path(spec.keyed(spec.manifest_path))
    if not manifest.exists():
        raise RuntimeError("Split manifest not found. Build the datasets before calling start_run().")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.runs_dir) / f"{ts}_{name}_seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (run_dir / "env.json").write_text(json.dumps(env_info(), indent=2))
    (run_dir / "data_spec.json").write_text(json.dumps({
        "spec": asdict(spec),
        "spec_key": spec.key(),
        "manifest_sha256_12": json.loads(manifest.read_text())["sha256_12"],
    }, indent=2))
    shutil.copy(manifest, run_dir / "splits_manifest.json")  # run is self-contained
    return run_dir
