"""Learning-rate schedule and class weights.

Reported: 5-epoch warmup then cosine decay to 1e-6, peak 3e-4, max 100 epochs.
Unreported (reconstruction): the schedule is stepped per iteration, warmup is linear from peak/warmup_steps, and
the cosine horizon is ``max_epochs`` (it does not shorten when early stopping triggers).
"""
from __future__ import annotations

import math

import numpy as np
import torch


def lr_at_step(step: int, steps_per_epoch: int, warmup_epochs: int, max_epochs: int, peak_lr: float, min_lr: float) -> float:
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = max_epochs * steps_per_epoch
    if warmup_steps and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


def make_scheduler(optimizer: torch.optim.Optimizer, steps_per_epoch: int, warmup_epochs: int, max_epochs: int, peak_lr: float, min_lr: float):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_at_step(s, steps_per_epoch, warmup_epochs, max_epochs, peak_lr, min_lr) / peak_lr
    )


def class_weights(train_counts: dict[str, int] | list[int] | np.ndarray, class_names: tuple[str, ...] | None = None) -> torch.Tensor:
    """weight[c] = N_train / (K * n_c). Raises if a class has no training sample."""
    if isinstance(train_counts, dict):
        assert class_names is not None
        counts = np.array([train_counts[c] for c in class_names], dtype=np.float64)
    else:
        counts = np.asarray(train_counts, dtype=np.float64)
    if (counts <= 0).any():
        raise ValueError(f"every class needs at least one training sample, got {counts.tolist()}")
    return torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)
