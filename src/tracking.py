from pathlib import Path

import mlflow
import tensorflow as tf

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "neurovision-mri"

def setup(experiment: str = EXPERIMENT) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment)

class MlflowEpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, prefix: str,  offset : int = 0):
        super().__init__()
        self.prefix = prefix
        self.offset = offset

    def on_epoch_end(self, epoch, logs=None):
        metrics = {f"{self.prefix}_{k}": float(v) for k, v in (logs or {}).items()}
        mlflow.log_metrics(metrics, step=self.offset + epoch)

def log_run_folder(run_dir: Path) -> None:
    mlflow.log_artifacts(str(run_dir), artifact_path="run")
