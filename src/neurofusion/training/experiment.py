"""Experiment orchestration: train on train/val, then evaluate on test ONCE after training."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from neurofusion.constants import CLASS_NAMES, MANUSCRIPT_TABLE1
from neurofusion.data.dataset import ManifestDataset, make_loader
from neurofusion.data.manifest import validate_manifest
from neurofusion.data.splits import audit_split
from neurofusion.data.transforms import AugmentationConfig, build_eval_transform, build_train_transform
from neurofusion.evaluation.metrics import classification_report
from neurofusion.evaluation.predict import predict_loader, save_predictions
from neurofusion.training.schedule import class_weights
from neurofusion.training.trainer import TrainConfig, Trainer
from neurofusion.utils.env import environment_info
from neurofusion.utils.hashing import dataset_fingerprint, split_fingerprint
from neurofusion.utils.seed import set_seed

ModelBuilder = Callable[[], nn.Module]


def run_experiment(
    manifest: pd.DataFrame,
    image_root: str | Path,
    out_dir: str | Path,
    model_builder: ModelBuilder,
    model_name: str,
    train_cfg: TrainConfig | None = None,
    aug_cfg: AugmentationConfig | None = None,
    model_config: dict | None = None,
    device: str | None = None,
    max_epochs: int | None = None,
    resume: bool = True,
    evaluate_test: bool = True,
    pretrained_provenance: dict | None = None,
) -> dict:
    """Train, select the best epoch on VALIDATION loss, then (optionally) predict on test and validation.

    Returns a record that is also written to ``run_record.json``. Test predictions are written only after the
    best checkpoint is fixed. ``max_epochs`` is for smoke tests; a run limited that way is flagged.
    """
    train_cfg = train_cfg or TrainConfig()
    aug_cfg = aug_cfg or AugmentationConfig()
    validate_manifest(manifest, require_split=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(train_cfg.seed)

    tr = manifest[manifest["split"] == "train"]
    va = manifest[manifest["split"] == "val"]
    te = manifest[manifest["split"] == "test"]
    if tr.empty or va.empty:
        raise ValueError("train and val splits must be non-empty")
    train_ds = ManifestDataset(tr, image_root, build_train_transform(aug_cfg))
    val_ds = ManifestDataset(va, image_root, build_eval_transform())
    counts = tr["label"].value_counts().reindex(CLASS_NAMES, fill_value=0).to_dict()
    weights = class_weights(counts, CLASS_NAMES)
    train_loader = make_loader(train_ds, train_cfg.batch_size, True, train_cfg.seed, train_cfg.num_workers)
    val_loader = make_loader(val_ds, train_cfg.batch_size, False, train_cfg.seed, train_cfg.num_workers)

    model = model_builder()
    provenance = {
        "model_name": model_name,
        "model_config": model_config,
        "train_config": train_cfg.as_dict(),
        "augmentation_config": aug_cfg.as_dict(),
        "class_weights": dict(zip(CLASS_NAMES, weights.tolist())),
        "dataset_fingerprint": dataset_fingerprint(manifest),
        "split_fingerprint": split_fingerprint(manifest),
        "split_audit": audit_split(manifest, MANUSCRIPT_TABLE1),
        "environment": environment_info(),
        "pretrained": pretrained_provenance,
        "seed": train_cfg.seed,
        "status": "NEW EXPERIMENT (not verified to reproduce the manuscript's original run)",
    }
    trainer = Trainer(model, train_loader, val_loader, train_cfg, out_dir, weights, device, model_config, extra_provenance={"split_fingerprint": provenance["split_fingerprint"]})
    if resume:
        trainer.resume()
    summary = trainer.fit(max_epochs)
    summary["truncated_for_smoke_test"] = bool(max_epochs and max_epochs < train_cfg.max_epochs and not summary["stopped_early"])
    record = {**provenance, "train_summary": summary}
    if trainer.best_ckpt.is_file():
        trainer.load_best()
    record["best_checkpoint"] = str(trainer.best_ckpt)
    if evaluate_test and len(te):
        te_loader = make_loader(ManifestDataset(te, image_root, build_eval_transform()), train_cfg.batch_size, False)
        pred_test = predict_loader(model, te_loader, trainer.device)
        save_predictions(pred_test, out_dir / "predictions_test.csv", {"split": "test", "model": model_name, "config_hash": trainer.cfg_hash})
        record["test_report"] = classification_report(pred_test)
    pred_val = predict_loader(model, val_loader, trainer.device)
    save_predictions(pred_val, out_dir / "predictions_val.csv", {"split": "val", "model": model_name})
    record["val_report"] = classification_report(pred_val)
    (out_dir / "run_record.json").write_text(json.dumps(record, indent=2, default=str))
    return record


def run_repeated_seeds(manifest, image_root, out_root, model_builder, model_name, seeds: list[int], **kw) -> pd.DataFrame:
    """Same data and split, different training seeds (training variability; NOT test-set uncertainty)."""
    rows = []
    base_cfg = kw.pop("train_cfg", None) or TrainConfig()
    for s in seeds:
        cfg = TrainConfig(**{**base_cfg.as_dict(), "seed": s})
        rec = run_experiment(manifest, image_root, Path(out_root) / f"seed_{s}", model_builder, model_name, train_cfg=cfg, **kw)
        t = rec.get("test_report")
        rows.append({"seed": s, "accuracy": t["accuracy"] if t else None, "macro_f1": t["macro"]["f1"] if t else None,
                     "best_epoch": rec["train_summary"]["best_epoch"]})
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_root) / "seeds_summary.csv", index=False)
    return df


def run_cross_validation(folds: list[pd.DataFrame], image_root, out_root, model_builder, model_name, **kw) -> pd.DataFrame:
    rows = []
    for i, f in enumerate(folds):
        rec = run_experiment(f.drop(columns=[c for c in ("fold",) if c in f.columns]), image_root, Path(out_root) / f"fold_{i}", model_builder, model_name, **kw)
        t = rec.get("test_report")
        rows.append({"fold": i, "n_train": int((f["split"] == "train").sum()), "n_val": int((f["split"] == "val").sum()),
                     "n_test": int((f["split"] == "test").sum()), "accuracy": t["accuracy"] if t else None,
                     "macro_f1": t["macro"]["f1"] if t else None, "macro_auc": t["macro"]["auc"] if t else None})
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_root) / "cv_summary.csv", index=False)
    return df


def cuda_free() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
