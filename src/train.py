from src.config import Config
from src.pipeline import DataSpec, get_datasets
from src.repro import set_seed, start_run

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
    run_dir = start_run(cfg, spec, name="baseline")
    print("Run dir:", run_dir)

    # model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, class_weight=class_weight, ...)


if __name__ == "__main__":
    main()