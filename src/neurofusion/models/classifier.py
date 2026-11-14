"""Fusion classifier (proposed model, ablations and controls) and prediction ensemble."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from neurofusion.constants import CLASS_NAMES, NUM_CLASSES
from neurofusion.models.backbones import (
    COATNET0,
    DINOV2_S14,
    BackboneSpec,
    create_backbone,
    set_trainable,
)
from neurofusion.models.fusion import ConcatFusion, MLPControl, SEGate, SumFusion


@dataclass
class ModelConfig:
    name: str = "proposed"
    branches: str = "both"  # "both" | "a" (CoAtNet only) | "b" (DINOv2 only)
    fusion: str = "concat"  # "concat" | "sum"
    attention: str = "se"  # "se" | "none" | "mlp_control"
    reduction: int = 8
    hidden1: int = 512
    hidden2: int = 128
    dropout1: float = 0.40
    dropout2: float = 0.30
    num_classes: int = NUM_CLASSES
    pretrained: bool = True
    trainable_a: str = "full"
    trainable_b: str = "full"
    mlp_control_mode: str = "residual"

    def as_dict(self) -> dict:
        return asdict(self)


class ClassifierHead(nn.Module):
    """Manuscript Table 2: Dense(D->512) ReLU Dropout(0.4) Dense(512->128) ReLU Dropout(0.3) Dense(128->4)."""

    def __init__(self, in_dim: int, h1: int, h2: int, d1: float, d2: float, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(d1),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(d2),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits; softmax is applied only for reporting


class FusionClassifier(nn.Module):
    """Branch A = CoAtNet-0 (768-d), branch B = DINOv2 ViT-S/14 (384-d). Returns logits."""

    def __init__(self, cfg: ModelConfig, branch_a: nn.Module | None, branch_b: nn.Module | None, dim_a: int = 768, dim_b: int = 384):
        super().__init__()
        self.cfg = cfg
        self.class_names = CLASS_NAMES
        use_a, use_b = cfg.branches in {"both", "a"}, cfg.branches in {"both", "b"}
        if use_a and branch_a is None or use_b and branch_b is None:
            raise ValueError("a required branch module was not supplied")
        self.branch_a = branch_a if use_a else None
        self.branch_b = branch_b if use_b else None
        if cfg.branches == "both":
            self.fusion: nn.Module | None = (
                ConcatFusion(dim_a, dim_b) if cfg.fusion == "concat" else SumFusion(dim_a, dim_b)
            )
            fused = self.fusion.out_dim
        else:
            self.fusion = None
            fused = dim_a if use_a else dim_b
        if cfg.attention == "se":
            self.attention: nn.Module = SEGate(fused, cfg.reduction)
        elif cfg.attention == "mlp_control":
            self.attention = MLPControl(fused, cfg.reduction, cfg.mlp_control_mode)
        elif cfg.attention == "none":
            self.attention = nn.Identity()
        else:
            raise ValueError(f"unknown attention {cfg.attention!r}")
        self.head = ClassifierHead(fused, cfg.hidden1, cfg.hidden2, cfg.dropout1, cfg.dropout2, cfg.num_classes)
        self.fused_dim = fused

    def features(self, x: torch.Tensor) -> torch.Tensor:
        a = self.branch_a(x) if self.branch_a is not None else None
        b = self.branch_b(x) if self.branch_b is not None else None
        z = self.fusion(a, b) if self.fusion is not None else (a if a is not None else b)
        return self.attention(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class PredictionEnsemble(nn.Module):
    """Average of softmax probabilities from separate classifiers (new extension). Returns log of the mean
    probability so that ``softmax(output)`` recovers the averaged probabilities."""

    def __init__(self, models: list[nn.Module], weights: list[float] | None = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        w = torch.tensor(weights if weights else [1.0] * len(models), dtype=torch.float32)
        self.register_buffer("w", w / w.sum())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.stack([torch.softmax(m(x), dim=1) for m in self.models])
        mean = (probs * self.w.view(-1, 1, 1)).sum(0)
        return torch.log(mean.clamp_min(1e-12))


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "total_millions": round(total / 1e6, 4), "trainable_millions": round(trainable / 1e6, 4)}


def build_fusion_model(cfg: ModelConfig, spec_a: BackboneSpec = COATNET0, spec_b: BackboneSpec = DINOV2_S14) -> FusionClassifier:
    """Instantiate the real backbones (timm). Weights are downloaded when ``cfg.pretrained``."""
    a = create_backbone(spec_a, cfg.pretrained) if cfg.branches in {"both", "a"} else None
    b = create_backbone(spec_b, cfg.pretrained) if cfg.branches in {"both", "b"} else None
    model = FusionClassifier(cfg, a, b, spec_a.expected_dim, spec_b.expected_dim)
    if a is not None:
        set_trainable(a, cfg.trainable_a)
    if b is not None:
        set_trainable(b, cfg.trainable_b)
    return model
