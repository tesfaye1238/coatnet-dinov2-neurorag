"""Backbone adapters with dimension verification. Never silently substitute another backbone."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import timm
import torch
from torch import nn


class BackboneDimensionError(RuntimeError):
    """The instantiated backbone does not produce the feature size the manuscript reports."""


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    timm_name: str
    pretrained_tag: str
    expected_dim: int
    img_size: int = 224
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# Manuscript Table 2: "CoAtNet-0 ... global average pooled 768-d" and "DINOv2 ViT-S/14 ... CLS 384-d".
# The manuscript does not name a library. timm's coatnet_0_rw_224 is the CoAtNet-0 variant with ImageNet weights
# available in timm; its identity with the authors' CoAtNet-0 is NOT verified (see docs/implementation_decisions.md).
COATNET0 = BackboneSpec(
    "coatnet_0", "coatnet_0_rw_224", "sw_in1k", 768,
    note="timm coatnet_0_rw_224.sw_in1k; num_classes=0 gives global-average-pooled 768-d features",
)
DINOV2_S14 = BackboneSpec(
    "dinov2_vits14", "vit_small_patch14_dinov2", "lvd142m", 384,
    note="timm vit_small_patch14_dinov2.lvd142m (self-supervised), img_size=224 (position embeddings interpolated); CLS token",
)


def create_backbone(spec: BackboneSpec, pretrained: bool = True) -> nn.Module:
    kwargs = {"num_classes": 0, "pretrained": pretrained}
    if "dinov2" in spec.timm_name:
        kwargs["img_size"] = spec.img_size
    name = f"{spec.timm_name}.{spec.pretrained_tag}" if pretrained else spec.timm_name
    model = timm.create_model(name, **kwargs)
    verify_backbone_dim(model, spec)
    return model


@torch.no_grad()
def verify_backbone_dim(model: nn.Module, spec: BackboneSpec) -> int:
    was_training = model.training
    model.eval()
    out = model(torch.zeros(1, 3, spec.img_size, spec.img_size))
    model.train(was_training)
    if out.ndim != 2 or out.shape[1] != spec.expected_dim:
        raise BackboneDimensionError(
            f"{spec.name}: expected a pooled [B, {spec.expected_dim}] feature vector, got {tuple(out.shape)}. "
            "Do not add an ad-hoc projection silently; if one is needed, document it as a methodological change."
        )
    return out.shape[1]


def set_trainable(module: nn.Module, mode: str = "full", last_n: int = 0) -> str:
    """``full`` (all params train), ``frozen`` (none) or ``last_n`` (the final ``last_n`` parameter-bearing direct children train).

    The manuscript does not state the freezing policy. ``full`` is a reconstruction default.
    """
    if mode == "full":
        for p in module.parameters():
            p.requires_grad = True
    elif mode == "frozen":
        for p in module.parameters():
            p.requires_grad = False
    elif mode == "last_n":
        for p in module.parameters():
            p.requires_grad = False
        kids = [c for c in module.children() if any(True for _ in c.parameters())]  # parameter-bearing children only
        for child in kids[-max(last_n, 1):]:
            for p in child.parameters():
                p.requires_grad = True
    else:
        raise ValueError(f"unknown trainable mode {mode!r}")
    return mode


def pretrained_provenance(spec: BackboneSpec) -> dict:
    """Identifier of the weights requested. The hub commit is not pinned by timm; recorded as unpinned."""
    try:
        cfg = timm.models.get_pretrained_cfg(f"{spec.timm_name}.{spec.pretrained_tag}")
        hub = getattr(cfg, "hf_hub_id", None)
    except Exception:  # noqa: BLE001
        hub = None
    return {**spec.as_dict(), "hf_hub_id": hub, "hub_revision": "unpinned (latest main at download time)", "timm_version": timm.__version__}
