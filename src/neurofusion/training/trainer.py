"""Training loop. The trainer takes only train and validation loaders; test data cannot enter the loop."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from neurofusion.training.checkpoint import load_checkpoint, save_checkpoint
from neurofusion.training.schedule import make_scheduler
from neurofusion.utils.hashing import config_hash


@dataclass
class TrainConfig:
    # REPORTED (manuscript section 5, Table 3)
    batch_size: int = 32
    peak_lr: float = 3e-4
    weight_decay: float = 5e-4
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    max_epochs: int = 100
    patience: int = 10
    # RECONSTRUCTION CHOICES (unreported)
    amp: bool = False
    grad_clip: float | None = None
    seed: int = 0
    num_workers: int = 0
    min_delta: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainConfig,
        out_dir: str | Path,
        class_weight: torch.Tensor | None = None,
        device: str | torch.device | None = None,
        model_config: dict | None = None,
        extra_provenance: dict | None = None,
    ):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.criterion = nn.CrossEntropyLoss(weight=class_weight.to(self.device) if class_weight is not None else None)
        # validation loss is reported unweighted-by-class-weight? The manuscript reports weighted CE as the loss.
        # We monitor the same weighted loss on validation for early stopping (a reconstruction choice).
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=cfg.peak_lr, weight_decay=cfg.weight_decay)
        self.scheduler = make_scheduler(self.optimizer, len(train_loader), cfg.warmup_epochs, cfg.max_epochs, cfg.peak_lr, cfg.min_lr)
        self.use_amp = cfg.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.model_config = model_config or {}
        self.cfg_hash = config_hash({"train": cfg.as_dict(), "model": self.model_config})
        self.extra = extra_provenance or {}
        self.history: list[dict] = []
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.bad_epochs = 0

    @property
    def last_ckpt(self) -> Path:
        return self.out_dir / "last.pt"

    @property
    def best_ckpt(self) -> Path:
        return self.out_dir / "best.pt"

    def resume(self) -> bool:
        if not self.last_ckpt.is_file():
            return False
        ck = load_checkpoint(self.last_ckpt, map_location=self.device)
        self.model.load_state_dict(ck["model_state"])
        self.optimizer.load_state_dict(ck["optimizer_state"])
        self.scheduler.load_state_dict(ck["scheduler_state"])
        if ck.get("scaler_state") and self.use_amp:
            self.scaler.load_state_dict(ck["scaler_state"])
        self.start_epoch = ck["epoch"] + 1
        self.best_val_loss = ck["best_val_loss"] if ck["best_val_loss"] is not None else float("inf")
        self.best_epoch = ck["best_epoch"] if ck["best_epoch"] is not None else -1
        self.bad_epochs = ck["epochs_without_improvement"]
        hist = self.out_dir / "history.csv"
        if hist.is_file():
            self.history = pd.read_csv(hist).to_dict("records")[: self.start_epoch]
        return True

    def _train_epoch(self) -> tuple[float, float]:
        self.model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y, _ in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        return loss_sum / max(total, 1), correct / max(total, 1)

    def _save(self, path: Path, epoch: int) -> None:
        save_checkpoint(
            path, self.model, epoch=epoch, optimizer=self.optimizer, scheduler=self.scheduler,
            scaler=self.scaler if self.use_amp else None, best_val_loss=self.best_val_loss, best_epoch=self.best_epoch,
            epochs_without_improvement=self.bad_epochs, model_config=self.model_config, run_config=self.cfg.as_dict(),
            config_hash=self.cfg_hash, extra=self.extra,
        )

    def fit(self, max_epochs: int | None = None) -> dict:
        """Train until ``max_epochs`` or early stopping. Best checkpoint = lowest VALIDATION loss.

        ``max_epochs`` may stop early for smoke tests; it does not alter the LR schedule horizon."""
        stop_at = min(max_epochs or self.cfg.max_epochs, self.cfg.max_epochs)
        stopped_early = False
        for epoch in range(self.start_epoch, stop_at):
            t0 = time.perf_counter()
            tr_loss, tr_acc = self._train_epoch()
            va_loss, va_acc = evaluate_loss(self.model, self.val_loader, self.criterion, self.device)
            improved = va_loss < self.best_val_loss - self.cfg.min_delta
            if improved:
                self.best_val_loss, self.best_epoch, self.bad_epochs = va_loss, epoch, 0
            else:
                self.bad_epochs += 1
            self.history.append(
                {
                    "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc,
                    "lr": self.optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - t0,
                }
            )
            pd.DataFrame(self.history).to_csv(self.out_dir / "history.csv", index=False)
            self._save(self.last_ckpt, epoch)
            if improved:
                self._save(self.best_ckpt, epoch)
            if self.bad_epochs >= self.cfg.patience:
                stopped_early = True
                break
        summary = {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "epochs_run": len(self.history),
            "stopped_early": stopped_early,
            "config_hash": self.cfg_hash,
            "history_note": "history contains TRAIN and VALIDATION curves only; no test data was evaluated during training",
        }
        (self.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    def load_best(self) -> None:
        ck = load_checkpoint(self.best_ckpt, map_location=self.device)
        self.model.load_state_dict(ck["model_state"])
