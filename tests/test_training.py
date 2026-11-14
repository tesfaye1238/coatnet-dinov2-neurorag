import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from conftest import TinyBranch, TinyViT
from neurofusion.constants import CLASS_NAMES
from neurofusion.data import splits as S
from neurofusion.data.dataset import ManifestDataset, make_loader
from neurofusion.data.transforms import build_eval_transform
from neurofusion.models.classifier import FusionClassifier, ModelConfig
from neurofusion.training.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from neurofusion.training.experiment import run_cross_validation, run_experiment, run_repeated_seeds
from neurofusion.training.schedule import class_weights, lr_at_step
from neurofusion.training.trainer import TrainConfig, Trainer, evaluate_loss
from neurofusion.utils.hashing import split_fingerprint
from neurofusion.utils.seed import set_seed


def test_lr_schedule_shape():
    spe, w, E, peak, lo = 10, 5, 100, 3e-4, 1e-6
    first = lr_at_step(0, spe, w, E, peak, lo)
    end_warm = lr_at_step(w * spe - 1, spe, w, E, peak, lo)
    last = lr_at_step(E * spe - 1, spe, w, E, peak, lo)
    assert first == pytest.approx(peak / (w * spe)) and end_warm == pytest.approx(peak)
    assert last == pytest.approx(lo, rel=1e-6)
    assert lr_at_step(w * spe, spe, w, E, peak, lo) <= peak
    vals = [lr_at_step(s, spe, w, E, peak, lo) for s in range(w * spe, E * spe)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))  # monotone cosine decay


def test_class_weight_formula_and_zero_class():
    w = class_weights([100, 50, 25, 25])
    assert w.tolist() == pytest.approx([200 / (4 * 100), 200 / (4 * 50), 200 / (4 * 25), 200 / (4 * 25)])
    assert class_weights({"glioma": 1195, "meningioma": 1196, "notumor": 1194, "pituitary": 1197}, CLASS_NAMES).shape == (4,)
    with pytest.raises(ValueError):
        class_weights([10, 0, 5, 5])


def tiny_builder():
    def f():
        return FusionClassifier(ModelConfig(pretrained=False, hidden1=16, hidden2=8), TinyBranch(24), TinyViT(16), 24, 16)
    return f


@pytest.fixture
def split_manifest(synth_manifest):
    return S.group_stratified_split(synth_manifest, 0.2, 0.25, seed=0)


def fast_cfg(**kw):
    return TrainConfig(batch_size=8, peak_lr=1e-2, warmup_epochs=1, max_epochs=6, patience=3, seed=0, **kw)


def test_checkpoint_stores_and_verifies_class_mapping(tmp_path):
    m = nn.Linear(2, 2)
    p = save_checkpoint(tmp_path / "c.pt", m, epoch=3, config_hash="abc")
    ck = load_checkpoint(p)
    assert ck["class_to_index"] == {c: i for i, c in enumerate(CLASS_NAMES)} and ck["epoch"] == 3
    with pytest.raises(CheckpointError, match="class mapping"):
        load_checkpoint(p, expected_classes=("a", "b"))
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "missing.pt")


def test_run_experiment_smoke_and_artifacts(tmp_path, synth_root, split_manifest):
    rec = run_experiment(split_manifest, synth_root, tmp_path / "run", tiny_builder(), "tiny", fast_cfg(), max_epochs=3, device="cpu")
    out = tmp_path / "run"
    for f in ("history.csv", "best.pt", "last.pt", "train_summary.json", "predictions_test.csv", "predictions_val.csv", "run_record.json"):
        assert (out / f).is_file(), f
    hist = pd.read_csv(out / "history.csv")
    assert list(hist.columns[:5]) == ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    assert "test" not in " ".join(hist.columns)  # no per-epoch test curves
    assert rec["train_summary"]["truncated_for_smoke_test"] is True
    assert "NEW EXPERIMENT" in rec["status"]
    assert rec["split_fingerprint"] == split_fingerprint(split_manifest)
    saved = json.loads((out / "run_record.json").read_text())
    assert saved["model_config"] is None or isinstance(saved["model_config"], dict)
    assert saved["test_report"]["n"] == int((split_manifest["split"] == "test").sum())
    assert 0 <= saved["test_report"]["accuracy"] <= 1


def test_best_checkpoint_selected_on_validation_loss(tmp_path, synth_root, split_manifest):
    run_experiment(split_manifest, synth_root, tmp_path / "r", tiny_builder(), "tiny", fast_cfg(), max_epochs=4, device="cpu")
    hist = pd.read_csv(tmp_path / "r" / "history.csv")
    ck = load_checkpoint(tmp_path / "r" / "best.pt")
    assert ck["best_epoch"] == int(hist.loc[hist["val_loss"].idxmin(), "epoch"])


def test_trainer_has_no_test_loader_and_resume_continues(tmp_path, synth_root, split_manifest):
    set_seed(0)
    mk = lambda s: make_loader(ManifestDataset(split_manifest[split_manifest["split"] == s], synth_root, build_eval_transform()), 8, s == "train")  # noqa: E731
    cfg = fast_cfg()
    tr = Trainer(tiny_builder()(), mk("train"), mk("val"), cfg, tmp_path / "t", device="cpu")
    tr.fit(max_epochs=2)
    assert len(tr.history) == 2
    tr2 = Trainer(tiny_builder()(), mk("train"), mk("val"), cfg, tmp_path / "t", device="cpu")
    assert tr2.resume() is True and tr2.start_epoch == 2 and len(tr2.history) == 2
    tr2.fit(max_epochs=4)
    assert [h["epoch"] for h in tr2.history] == [0, 1, 2, 3]
    import inspect

    assert "test" not in " ".join(inspect.signature(Trainer.__init__).parameters)


def test_early_stopping(tmp_path, synth_root, split_manifest):
    mk = lambda s: make_loader(ManifestDataset(split_manifest[split_manifest["split"] == s], synth_root, build_eval_transform()), 8, s == "train")  # noqa: E731
    cfg = TrainConfig(batch_size=8, peak_lr=1e-12, warmup_epochs=0, max_epochs=20, patience=2, min_delta=10.0)  # nothing can improve by 10
    tr = Trainer(tiny_builder()(), mk("train"), mk("val"), cfg, tmp_path / "es", device="cpu")
    s = tr.fit()
    assert s["stopped_early"] and s["epochs_run"] < 20


def test_run_experiment_requires_nonempty_train_val(tmp_path, synth_root, synth_manifest):
    m = S.group_stratified_split(synth_manifest, 0.0, 0.3, seed=0)  # val empty
    with pytest.raises(ValueError, match="non-empty"):
        run_experiment(m, synth_root, tmp_path / "x", tiny_builder(), "tiny", fast_cfg(), max_epochs=1, device="cpu")


def test_repeated_seeds_and_cv(tmp_path, synth_root, synth_manifest, split_manifest):
    df = run_repeated_seeds(split_manifest, synth_root, tmp_path / "seeds", tiny_builder(), "tiny", [0, 1], train_cfg=fast_cfg(), max_epochs=1, device="cpu")
    assert df["seed"].tolist() == [0, 1] and (tmp_path / "seeds" / "seeds_summary.csv").is_file()
    folds = S.kfold_splits(synth_manifest, k=3, seed=0)[:2]
    cv = run_cross_validation(folds, synth_root, tmp_path / "cv", tiny_builder(), "tiny", train_cfg=fast_cfg(), max_epochs=1, device="cpu")
    assert cv["n_test"].tolist() == [int((f["split"] == "test").sum()) for f in folds]


def test_evaluate_loss_runs(synth_root, split_manifest):
    loader = make_loader(ManifestDataset(split_manifest[split_manifest["split"] == "val"], synth_root, build_eval_transform()), 8, False)
    loss, acc = evaluate_loss(tiny_builder()(), loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert loss > 0 and 0 <= acc <= 1


def test_seed_reproducibility():
    set_seed(5)
    a = torch.randn(3)
    set_seed(5)
    assert torch.equal(a, torch.randn(3))
    assert np.random.rand() is not None
