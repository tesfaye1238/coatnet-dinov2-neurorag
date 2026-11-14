"""Name -> model factory for the proposed model, the reported ablations, new controls and baselines."""
from __future__ import annotations

from torch import nn

from neurofusion.models.baselines import BASELINES, build_baseline
from neurofusion.models.classifier import ModelConfig, build_fusion_model

# Reported ablations (manuscript Table 5)
ABLATIONS: dict[str, ModelConfig] = {
    "coatnet_only": ModelConfig(name="coatnet_only", branches="a", attention="none"),
    "dinov2_only": ModelConfig(name="dinov2_only", branches="b", attention="none"),
    "concat_no_attention": ModelConfig(name="concat_no_attention", attention="none"),
    "concat_se_r16": ModelConfig(name="concat_se_r16", reduction=16),
    "sum_se_r8": ModelConfig(name="sum_se_r8", fusion="sum"),
    "proposed": ModelConfig(name="proposed"),
}

# New controls (NOT in the manuscript)
EXTENSIONS: dict[str, ModelConfig] = {
    "concat_mlp_control": ModelConfig(name="concat_mlp_control", attention="mlp_control"),
}


def available_models() -> list[str]:
    return sorted(set(ABLATIONS) | set(EXTENSIONS) | set(BASELINES))


def build_model(name: str, pretrained: bool = True, **overrides) -> nn.Module:
    if name in ABLATIONS or name in EXTENSIONS:
        base = {**ABLATIONS, **EXTENSIONS}[name]
        cfg = ModelConfig(**{**base.as_dict(), "pretrained": pretrained, **overrides})
        return build_fusion_model(cfg)
    if name in BASELINES:
        return build_baseline(name, pretrained=pretrained)
    raise KeyError(f"unknown model {name!r}; known: {available_models()}")
