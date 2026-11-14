"""Baseline adapters. A missing baseline raises; it is never replaced by a similarly named model."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import timm
from torch import nn

from neurofusion.constants import NUM_CLASSES


class BaselineUnavailable(RuntimeError):
    """No verified implementation of this baseline is available."""


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    source: str  # "timm" | "torchvision" | "external"
    identifier: str
    weights: str
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


BASELINES: dict[str, BaselineSpec] = {
    "GhostNetV2": BaselineSpec("GhostNetV2", "timm", "ghostnetv2_100", "in1k", "width multiplier 1.0; manuscript does not state the width"),
    "ConvNeXt-T": BaselineSpec("ConvNeXt-T", "timm", "convnext_tiny", "fb_in1k", "ImageNet-1k supervised; the manuscript does not state the pretraining set"),
    "Swin-Transformer-T": BaselineSpec("Swin-Transformer-T", "timm", "swin_tiny_patch4_window7_224", "ms_in1k", "patch 4, window 7, 224 input"),
    "MobileNetV2": BaselineSpec("MobileNetV2", "timm", "mobilenetv2_100", "ra_in1k", "width multiplier 1.0"),
    "ShuffleNetV2": BaselineSpec("ShuffleNetV2", "torchvision", "shufflenet_v2_x1_0", "IMAGENET1K_V1", "x1.0 width; the manuscript does not state the width"),
    "ESPNetV2": BaselineSpec(
        "ESPNetV2", "external", "unavailable", "none",
        "No ESPNetV2 classifier exists in timm or torchvision. A verified external implementation (provenance + licence) "
        "must be registered with register_external_baseline().",
    ),
}

_EXTERNAL: dict[str, nn.Module] = {}


def register_external_baseline(name: str, module: nn.Module, provenance: str, license_: str) -> None:
    """Register an externally sourced baseline. ``provenance`` and ``license_`` are mandatory."""
    if not provenance.strip() or not license_.strip():
        raise ValueError("provenance and license are required for an external baseline")
    _EXTERNAL[name] = module


def build_baseline(name: str, pretrained: bool = True, num_classes: int = NUM_CLASSES) -> nn.Module:
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; known: {sorted(BASELINES)}")
    spec = BASELINES[name]
    if spec.source == "external":
        if name in _EXTERNAL:
            return _EXTERNAL[name]
        raise BaselineUnavailable(f"{name}: {spec.note}")
    if spec.source == "timm":
        tag = f".{spec.weights}" if pretrained else ""
        return timm.create_model(f"{spec.identifier}{tag}", pretrained=pretrained, num_classes=num_classes)
    if spec.source == "torchvision":
        import torchvision

        weights = torchvision.models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.shufflenet_v2_x1_0(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise BaselineUnavailable(name)
