"""Checkpoint save/load. The class-to-index mapping is stored with every checkpoint and verified on load."""
from __future__ import annotations

from pathlib import Path

import torch

from neurofusion.constants import CLASS_NAMES


class CheckpointError(RuntimeError):
    pass


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler=None,
    best_val_loss: float | None = None,
    best_epoch: int | None = None,
    epochs_without_improvement: int = 0,
    model_config: dict | None = None,
    run_config: dict | None = None,
    config_hash: str | None = None,
    class_names: tuple[str, ...] = CLASS_NAMES,
    extra: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "model_config": model_config,
        "run_config": run_config,
        "config_hash": config_hash,
        "class_names": list(class_names),
        "class_to_index": {c: i for i, c in enumerate(class_names)},
        "rng_state": {"torch": torch.get_rng_state()},
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu", expected_classes: tuple[str, ...] = CLASS_NAMES) -> dict:
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint not found: {path}")
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if tuple(ck.get("class_names", ())) != tuple(expected_classes):
        raise CheckpointError(
            f"class mapping mismatch: checkpoint {ck.get('class_names')} vs expected {list(expected_classes)}"
        )
    return ck
