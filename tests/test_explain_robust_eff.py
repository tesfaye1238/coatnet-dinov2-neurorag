import json

import numpy as np
import pytest
import torch
from torch import nn

from neurofusion.data.manifest import build_manifest_from_folder
from neurofusion.evaluation import robustness as R
from neurofusion.evaluation.efficiency import benchmark_latency, count_flops, efficiency_report
from neurofusion.explainability.gradcam import (
    CAMExtractor,
    GradFlowError,
    export_attribution,
    normalise_cam,
    target_layer,
    vit_tokens_to_map,
)


def test_vit_token_reshape():
    t = torch.arange(2 * 17 * 3, dtype=torch.float32).reshape(2, 17, 3)
    m = vit_tokens_to_map(t)
    assert m.shape == (2, 3, 4, 4)
    assert torch.equal(m[0, :, 0, 0], t[0, 1, :])  # first patch token, class token dropped
    with pytest.raises(ValueError, match="square"):
        vit_tokens_to_map(torch.zeros(1, 1 + 15, 3))
    with pytest.raises(ValueError):
        vit_tokens_to_map(torch.zeros(3, 3))


@pytest.mark.parametrize("branch", ["a", "b"])
@pytest.mark.parametrize("method", ["gradcam", "gradcampp"])
def test_cam_shapes_and_gradient_reach_layer(tiny_fusion, branch, method):
    layer, reshape, _ = target_layer(tiny_fusion, branch)
    x = torch.randn(2, 3, 32, 32)
    with CAMExtractor(tiny_fusion, layer, method, reshape) as cam_fn:
        cam, logits, idx = cam_fn(x)
    assert cam.shape == (2, 32, 32) and logits.shape == (2, 4) and (cam >= 0).all()
    assert float(cam.sum()) > 0
    assert torch.equal(idx, logits.argmax(1))
    n = normalise_cam(cam)
    assert float(n.min()) >= 0 and float(n.max()) <= 1 + 1e-6


def test_last_block_output_has_no_patch_gradient_and_is_rejected(tiny_fusion):
    """Class-token pooling => the last block's OUTPUT carries no patch-token gradient; this must not silently yield a blank map."""
    with CAMExtractor(tiny_fusion, tiny_fusion.branch_b.blocks[-1], "gradcam", vit_tokens_to_map) as f:
        with pytest.raises(GradFlowError, match="identically zero"):
            f(torch.randn(1, 3, 32, 32))


def test_cam_target_class_changes_map(tiny_fusion):
    layer, _, _ = target_layer(tiny_fusion, "a")
    x = torch.randn(1, 3, 32, 32)
    with CAMExtractor(tiny_fusion, layer, "gradcam") as f:
        c0, _, _ = f(x, 0)
        c1, _, _ = f(x, 1)
    assert not torch.allclose(c0, c1)


def test_gradient_not_on_path_raises(tiny_fusion):
    stray = nn.Conv2d(3, 4, 1)  # not used in the forward pass of the model
    with CAMExtractor(tiny_fusion, stray, "gradcam") as f:
        with pytest.raises(GradFlowError):
            f(torch.randn(1, 3, 32, 32))


def test_unknown_branch_and_method(tiny_fusion):
    with pytest.raises(ValueError):
        target_layer(tiny_fusion, "c")
    with pytest.raises(ValueError):
        CAMExtractor(tiny_fusion, tiny_fusion.head, "scorecam")


def test_export_writes_raw_array_and_settings(tmp_path):
    img = np.random.rand(32, 32, 3).astype(np.float32)
    cam = np.random.rand(32, 32).astype(np.float32)
    meta = export_attribution(tmp_path, "case1", img, cam, {"method": "gradcampp"})
    assert (tmp_path / "case1.cam_raw.npy").is_file() and (tmp_path / "case1.png").is_file()
    assert np.allclose(np.load(tmp_path / "case1.cam_raw.npy"), cam)
    saved = json.loads((tmp_path / "case1.json").read_text())
    assert "not a segmentation mask" in saved["note"] and meta["method"] == "gradcampp"


# ---- robustness
@pytest.mark.parametrize("name", list(R.PERTURBATIONS))
def test_perturbations_preserve_shape_and_range(name):
    x = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    for lvl in R.SEVERITIES[name]:
        y = R.PERTURBATIONS[name](x, lvl, np.random.default_rng(1))
        assert y.shape == x.shape and y.min() >= -1e-6 and y.max() <= 1 + 1e-6, (name, lvl)


def test_severity_lists_match_supplement():
    assert R.SEVERITIES["gaussian_noise"] == (0.01, 0.02, 0.04, 0.06, 0.08)
    assert R.SEVERITIES["jpeg_compression"] == (70, 50, 30, 20, 10)
    assert all(len(v) == 5 for v in R.SEVERITIES.values())


def test_perturbation_strength_increases_with_severity():
    x = np.random.default_rng(0).random((64, 64, 3)).astype(np.float32)
    errs = [np.abs(R.gaussian_noise(x, s, np.random.default_rng(0)) - x).mean() for s in R.SEVERITIES["gaussian_noise"]]
    assert errs == sorted(errs)
    blur = [np.abs(R.gaussian_blur(x, s, np.random.default_rng(0)) - x).mean() for s in R.SEVERITIES["gaussian_blur"]]
    assert blur == sorted(blur)


def test_perturbations_are_seeded():
    x = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    a = R.bias_field(x, 0.6, np.random.default_rng(5))
    b = R.bias_field(x, 0.6, np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_evaluate_robustness_table(synth_root, tiny_fusion):
    df = build_manifest_from_folder(synth_root, "s")
    res = R.evaluate_robustness(tiny_fusion, df.iloc[:12], synth_root, perturbations=["gaussian_noise", "jpeg_compression"], batch_size=6)
    assert len(res) == 1 + 2 * 5 and res.iloc[0]["perturbation"] == "clean"
    assert res["n"].eq(12).all() and res["accuracy"].between(0, 1).all()


# ---- efficiency
def tiny():
    return nn.Sequential(nn.Conv2d(3, 4, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))


def test_flop_count_matches_conv_and_linear_formula():
    m = nn.Sequential(nn.Conv2d(3, 4, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))
    f = count_flops(m, (1, 3, 8, 8))
    conv = 2 * (3 * 3 * 3) * 4 * (6 * 6)
    lin = 2 * 4 * 2
    assert f["flops"] == conv + lin and f["gmacs"] == pytest.approx(f["gflops"] / 2)


def test_latency_report_fields_and_no_energy_claim():
    r = benchmark_latency(tiny(), (1, 3, 16, 16), warmup=1, iters=3, device="cpu")
    assert r["latency_ms_median"] > 0 and r["throughput_img_per_s"] > 0 and r["energy"] == "not measured"
    assert r["peak_gpu_memory_mb"] is None
    rep = efficiency_report(tiny(), iters=2, device="cpu")
    assert rep["parameters"]["total"] == sum(p.numel() for p in tiny().parameters())
