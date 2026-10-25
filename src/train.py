import json
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src import tracking
from src.config import Config
from src.EpochTimer import EpochTimer
from src.pipeline import DataSpec, get_datasets, get_test_ds
from src.repro import env_info, set_seed, start_run


def build_model(cfg: Config) -> tuple[tf.keras.Model, tf.keras.Model]:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*cfg.img_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.layers.Input((*cfg.img_size, 3))
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0)(inputs)  # [0,1] -> [-1,1] for MobileNetV2
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(cfg.dropout)(x)
    outputs = tf.keras.layers.Dense(len(cfg.classes), activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def compile_model(model: tf.keras.Model, lr: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def fit_stage(model, train_ds, val_ds, epochs, class_weight, ckpt_path, prefix, offset, extra_cb=()):
    timer = EpochTimer()
    callbacks = [
        timer,
        tracking.MLflowEpochLogger(prefix, offset),
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt_path), monitor="val_loss", save_best_only=True
        ),
        *extra_cb,
    ]
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    return {
        "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()},
        "epoch_times_s": [float(t) for t in timer.times],
        "total_s": float(sum(timer.times)),
    }


def evaluate_test(model, test_ds, classes, run_dir: Path) -> dict:
    loss, acc = model.evaluate(test_ds, verbose=0)
    y_true, y_pred = [], []
    for x, y in test_ds:
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(model.predict(x, verbose=0), axis=1).tolist())

    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()
    (run_dir / "classification_report.json").write_text(json.dumps(report, indent=2))
    (run_dir / "confusion_matrix.json").write_text(json.dumps(cm))

    for c in classes:
        mlflow.log_metric(f"test_recall_{c}", report[c]["recall"])
        mlflow.log_metric(f"test_f1_{c}", report[c]["f1-score"])
    mlflow.log_metric("test_macro_f1", report["macro avg"]["f1-score"])
    return {"loss": float(loss), "accuracy": float(acc)}


def main() -> None:
    cfg = Config()
    set_seed(cfg)  # before any dataset or model is built

    spec = DataSpec(
        raw_dir=cfg.data_dir,
        classes=cfg.classes,
        val_split=cfg.val_frac,
        test_split=cfg.test_frac,
        target_size=cfg.img_size,
        cache_path=cfg.cache_path,
        manifest_path=cfg.manifest_path,
        seed=cfg.seed,
    )

    train_ds, val_ds, class_weight = get_datasets(
        spec, cfg.batch_size, strategy="auto" if cfg.use_class_weight else "none"
    )
    run_dir = start_run(cfg, spec, name="mobilenetv2")
    print("Run dir:", run_dir)

    tracking.setup()
    with mlflow.start_run(run_name=run_dir.name):
        mlflow.log_params(cfg.to_dict())
        mlflow.log_param("data_spec_key", spec.key())
        mlflow.set_tags({k: str(v) for k, v in env_info().items()})
        mlflow.set_tag("run_dir", str(run_dir))

        model, base = build_model(cfg)

        # ---- stage 1: train the head, backbone frozen ----
        compile_model(model, cfg.learning_rate)
        head = fit_stage(
            model, train_ds, val_ds, cfg.head_epochs, class_weight,
            run_dir / "best_head.keras", "head", offset=0,
        )

        # ---- stage 2: fine-tune the last N backbone layers ----
        base.trainable = True
        for layer in base.layers[: -cfg.finetune_layers]:
            layer.trainable = False
        for layer in base.layers:  # keep BatchNorm statistics frozen
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        compile_model(model, cfg.finetune_lr)
        finetune = fit_stage(
            model, train_ds, val_ds, cfg.finetune_epochs, class_weight,
            run_dir / "model.keras", "finetune", offset=cfg.head_epochs,
            extra_cb=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=2
                ),
            ],
        )

        # ---- pick the best stage by val_loss ----
        head_best = min(head["history"]["val_loss"])
        ft_best = min(finetune["history"]["val_loss"])
        best_stage = "finetune" if ft_best <= head_best else "head"
        best_path = run_dir / ("model.keras" if best_stage == "finetune" else "best_head.keras")
        mlflow.log_param("best_stage", best_stage)
        mlflow.log_metric("best_val_loss", min(head_best, ft_best))
        mlflow.log_metric("total_train_s", head["total_s"] + finetune["total_s"])

        (run_dir / "history.json").write_text(
            json.dumps({"best_stage": best_stage, "head": head, "finetune": finetune}, indent=2)
        )

        # ---- test set: used exactly once ----
        best_model = tf.keras.models.load_model(best_path)
        test_ds = get_test_ds(spec, cfg.batch_size)
        result = evaluate_test(best_model, test_ds, list(cfg.classes), run_dir)
        (run_dir / "test_result.json").write_text(json.dumps(result, indent=2))
        mlflow.log_metrics({"test_loss": result["loss"], "test_accuracy": result["accuracy"]})
        print("Test:", result)

        tracking.log_run_folder(run_dir)


if __name__ == "__main__":
    main()