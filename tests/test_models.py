import pytest
import torch
from conftest import TinyBranch
from torch import nn

from neurofusion.models.backbones import (
    COATNET0,
    DINOV2_S14,
    BackboneDimensionError,
    BackboneSpec,
    create_backbone,
    set_trainable,
    verify_backbone_dim,
)
from neurofusion.models.baselines import (
    BASELINES,
    BaselineUnavailable,
    build_baseline,
    register_external_baseline,
)
from neurofusion.models.classifier import FusionClassifier, ModelConfig, PredictionEnsemble, count_parameters
from neurofusion.models.factory import ABLATIONS, available_models, build_model
from neurofusion.models.fusion import MLPControl, SEGate, SumFusion


def make(cfg_kw, da=768, db=384, ba=None, bb=None):
    cfg = ModelConfig(pretrained=False, **cfg_kw)
    return FusionClassifier(cfg, ba or nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, da)),
                            bb or nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, db)), da, db)


def test_concat_dims_and_logits():
    m = make({})
    assert m.fused_dim == 1152
    assert m.attention.fc1.out_features == 144 and m.attention.fc2.out_features == 1152  # r=8
    out = m(torch.randn(5, 3, 16, 16))
    assert out.shape == (5, 4)
    assert torch.allclose(out.softmax(1).sum(1), torch.ones(5), atol=1e-5)  # head returns raw logits; softmax is applied only for reporting


def test_head_layout_matches_table2():
    h = make({}).head.net
    dims = [(m.in_features, m.out_features) for m in h if isinstance(m, nn.Linear)]
    assert dims == [(1152, 512), (512, 128), (128, 4)]
    drops = [m.p for m in h if isinstance(m, nn.Dropout)]
    assert drops == [0.40, 0.30]


def test_se_gate_range_and_gradient_flow():
    se = SEGate(1152, 8)
    z = torch.randn(4, 1152, requires_grad=True)
    g = se.gate(z)
    assert g.shape == z.shape and float(g.min()) >= 0 and float(g.max()) <= 1
    (se(z) ** 2).sum().backward()
    assert z.grad is not None and se.fc1.weight.grad is not None and se.fc2.weight.grad.abs().sum() > 0
    assert torch.allclose(se(z), z * g)


def test_se_does_not_collapse_feature_dimension():
    assert SEGate(64, 8)(torch.randn(3, 64)).shape == (3, 64)
    with pytest.raises(ValueError):
        SEGate(100, 8)


def test_variant_parameter_differences_match_formulae():
    base = count_parameters(make({"attention": "none"}))["total"]
    r8 = count_parameters(make({"reduction": 8}))["total"]
    r16 = count_parameters(make({"reduction": 16}))["total"]
    assert r8 - base == 1152 * 144 + 144 + 144 * 1152 + 1152
    assert r16 - base == 1152 * 72 + 72 + 72 * 1152 + 1152


def test_sum_fusion_projects_smaller_branch():
    m = make({"fusion": "sum"})
    assert isinstance(m.fusion, SumFusion) and m.fused_dim == 768
    assert isinstance(m.fusion.proj_a, nn.Identity) and isinstance(m.fusion.proj_b, nn.Linear)
    assert m(torch.randn(2, 3, 8, 8)).shape == (2, 4)


def test_single_branch_models():
    a = make({"branches": "a", "attention": "none"})
    b = make({"branches": "b", "attention": "none"})
    assert a.fused_dim == 768 and b.fused_dim == 384 and a.branch_b is None and b.branch_a is None
    assert a(torch.randn(2, 3, 8, 8)).shape == (2, 4) and b(torch.randn(2, 3, 8, 8)).shape == (2, 4)


def test_mlp_control_has_same_params_and_no_gating():
    se, ctrl = SEGate(1152, 8), MLPControl(1152, 8)
    assert sum(p.numel() for p in se.parameters()) == sum(p.numel() for p in ctrl.parameters())
    z = torch.randn(3, 1152)
    h = ctrl.fc2(torch.relu(ctrl.fc1(z)))
    assert torch.allclose(ctrl(z), z + h)  # additive, not multiplicative
    assert torch.allclose(MLPControl(8, 2, "replace")(torch.randn(1, 8)).sum() * 0, torch.zeros(()))


def test_prediction_ensemble_probabilities_average():
    a, b = make({"attention": "none"}), make({"attention": "none"})
    ens = PredictionEnsemble([a, b]).eval()
    x = torch.randn(3, 3, 8, 8)
    with torch.no_grad():
        p = torch.softmax(ens(x), 1)
        ref = (torch.softmax(a.eval()(x), 1) + torch.softmax(b.eval()(x), 1)) / 2
    assert torch.allclose(p, ref, atol=1e-5)


def test_backbone_dim_mismatch_raises_clear_error():
    spec = BackboneSpec("fake", "n/a", "x", 999, img_size=16)
    with pytest.raises(BackboneDimensionError, match="expected a pooled"):
        verify_backbone_dim(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 5)), spec)


def test_trainable_modes():
    m = TinyBranch(8)
    set_trainable(m, "frozen")
    assert not any(p.requires_grad for p in m.parameters())
    set_trainable(m, "last_n", 1)
    assert any(p.requires_grad for p in m.parameters())
    set_trainable(m, "full")
    assert all(p.requires_grad for p in m.parameters())
    with pytest.raises(ValueError):
        set_trainable(m, "bogus")


def test_real_backbones_have_reported_dims():
    a = create_backbone(COATNET0, pretrained=False)
    b = create_backbone(DINOV2_S14, pretrained=False)
    with torch.no_grad():
        assert a.eval()(torch.zeros(1, 3, 224, 224)).shape == (1, 768)
        assert b.eval()(torch.zeros(1, 3, 224, 224)).shape == (1, 384)


def test_real_proposed_model_dims_and_param_counts():
    m = build_model("proposed", pretrained=False)
    assert m.fused_dim == 1152
    n = count_parameters(m)
    assert 49.0 < n["total_millions"] < 49.6  # computed, not hard-coded
    noatt = count_parameters(build_model("concat_no_attention", pretrained=False))["total"]
    assert n["total"] - noatt == 333_072


def test_factory_covers_ablations_and_baselines():
    names = available_models()
    for k in ABLATIONS:
        assert k in names
    for k in BASELINES:
        assert k in names
    with pytest.raises(KeyError):
        build_model("resnet50")


def test_missing_baseline_is_not_substituted():
    with pytest.raises(BaselineUnavailable, match="ESPNetV2"):
        build_baseline("ESPNetV2")
    with pytest.raises(ValueError):
        register_external_baseline("ESPNetV2", nn.Linear(1, 1), "", "")


def test_external_baseline_registration_requires_provenance():
    m = nn.Linear(1, 1)
    register_external_baseline("ESPNetV2", m, "https://example.org/espnetv2 @ commit abc", "MIT")
    assert build_baseline("ESPNetV2") is m
    from neurofusion.models import baselines

    baselines._EXTERNAL.clear()


def test_timm_baselines_instantiate_with_four_classes():
    for name in ("GhostNetV2", "ConvNeXt-T", "Swin-Transformer-T", "MobileNetV2"):
        m = build_baseline(name, pretrained=False).eval()
        with torch.no_grad():
            assert m(torch.zeros(1, 3, 224, 224)).shape == (1, 4), name
    m = build_baseline("ShuffleNetV2", pretrained=False).eval()
    with torch.no_grad():
        assert m(torch.zeros(1, 3, 224, 224)).shape == (1, 4)
