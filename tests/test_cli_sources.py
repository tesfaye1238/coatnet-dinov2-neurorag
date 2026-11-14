import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from neurofusion import cli
from neurofusion.data import sources as SRC
from neurofusion.data.figshare_import import import_figshare_mats, minmax_to_uint8, read_cjdata
from neurofusion.data.labels import LabelError, figshare_label
from neurofusion.utils.hashing import md5_file
from neurofusion.utils.publication import scan_files, scan_text_for_secrets


# ---- registry facts
def test_zenodo_record_is_flagged_as_not_a_dataset():
    z = SRC.get_source("zenodo_7786009_model_record")
    assert z.kind == "model_record_not_dataset" and z.contains_images is False
    assert "NONE is an image archive" in z.verification
    assert SRC.get_source("figshare_cheng_v8").contains_images is True
    assert SRC.get_source("br35h_kaggle").requires_authentication and SRC.get_source("sartaj_kaggle").requires_authentication
    with pytest.raises(SRC.SourceError):
        SRC.get_source("nope")


def test_figshare_registry_has_md5_for_every_file():
    f = SRC.FIGSHARE
    assert len(f.files) == 6 and all(len(x.md5) == 32 for x in f.files) and f.license.startswith("CC BY 4.0")
    assert "no no-tumor" in f.label_definitions.lower() or "No no-tumor" in f.label_definitions


def test_download_verifies_md5_and_removes_bad_file(tmp_path, monkeypatch):
    def fake(url, dest, timeout=60):
        dest.write_bytes(b"corrupt")

    monkeypatch.setattr(SRC, "_download", fake)
    with pytest.raises(SRC.SourceError, match="md5 mismatch"):
        SRC.download_figshare(tmp_path, only=["cvind.mat"], extract=False)
    assert not (tmp_path / "cvind.mat").exists()


def test_download_accepts_verified_file_without_redownload(tmp_path, monkeypatch):
    payload = b"hello"
    import hashlib

    rf = SRC.RemoteFile("cvind.mat", "https://x", 5, hashlib.md5(payload).hexdigest())  # noqa: S324
    monkeypatch.setattr(SRC, "FIGSHARE", SRC.SourceRecord(**{**SRC.FIGSHARE.__dict__, "files": (rf,)}))
    calls = []
    monkeypatch.setattr(SRC, "_download", lambda url, dest, timeout=60: (calls.append(url), dest.write_bytes(payload)))
    SRC.download_figshare(tmp_path, extract=False)
    SRC.download_figshare(tmp_path, extract=False)
    assert len(calls) == 1 and md5_file(tmp_path / "cvind.mat") == rf.md5


def test_zip_slip_rejected(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(SRC.SourceError, match="Unsafe path"):
        SRC.safe_extract_zip(z, tmp_path / "out")
    ok = tmp_path / "ok.zip"
    with zipfile.ZipFile(ok, "w") as zf:
        zf.writestr("a/b.txt", "x")
    assert (tmp_path / "out2" / "a" / "b.txt") in SRC.safe_extract_zip(ok, tmp_path / "out2")


# ---- figshare .mat import (synthetic HDF5 mimicking the cjdata structure)
def write_cj(path, label, pid, img):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        g = f.create_group("cjdata")
        g["label"] = np.array([[label]], dtype=np.float64)
        g["PID"] = np.array([[ord(c)] for c in pid], dtype=np.uint16)
        g["image"] = img.T.astype(np.int16)  # MATLAB column-major


def test_figshare_mat_import(tmp_path):
    rng = np.random.default_rng(0)
    for i, (lab, pid) in enumerate([(1, "100360"), (2, "100360"), (3, "999")]):
        write_cj(tmp_path / f"{i + 1}.mat", lab, pid, rng.integers(0, 3000, (16, 16)))
    df = import_figshare_mats(tmp_path, tmp_path / "png")
    assert df["label"].tolist() == ["meningioma", "glioma", "pituitary"]
    assert df["patient_id"].tolist() == ["100360", "100360", "999"] and df["patient_id_verified"].all()
    assert df["group_id"].iloc[0] == df["group_id"].iloc[1]  # same patient -> same group
    assert (df["grouping_method"] == "verified_patient_id").all()
    rec = read_cjdata(tmp_path / "1.mat")
    assert rec["image"].shape == (16, 16) and rec["pid"] == "100360"


def test_minmax_and_label_codes():
    a = minmax_to_uint8(np.array([[0, 50], [100, 200]]))
    assert a.min() == 0 and a.max() == 255 and a.dtype == np.uint8
    assert minmax_to_uint8(np.ones((2, 2))).sum() == 0
    assert figshare_label(2) == "glioma"
    with pytest.raises(LabelError):
        figshare_label(4)


# ---- publication checks
def test_secret_scan_finds_tokens_and_forbidden_files(tmp_path):
    (tmp_path / "a.py").write_text("tok = '" + "ghp_" + "A" * 36 + "'")
    (tmp_path / "ok.py").write_text("x = 1")
    (tmp_path / "w.pt").write_bytes(b"0")
    (tmp_path / "paper.pdf").write_bytes(b"%PDF")
    (tmp_path / ".env").write_text("A=1")
    (tmp_path / "u.md").write_text("https://user:" + "supersecret1" + "@example.org/x")  # synthetic fixture, assembled at runtime
    probs = scan_files(tmp_path, ["a.py", "ok.py", "w.pt", "paper.pdf", ".env", "u.md"])
    assert any(p["file"] == "a.py" and "possible secret" in p["problem"] for p in probs)
    assert any(p["file"] == "w.pt" and "forbidden" in p["problem"] for p in probs)
    assert any(p["file"] == "paper.pdf" for p in probs) and any(p["file"] == ".env" for p in probs) and any(p["file"] == "u.md" for p in probs)
    assert not any(p["file"] == "ok.py" for p in probs)
    assert scan_text_for_secrets("github_pat_" + "x" * 50) == ["github_pat_fine"]


# ---- CLI end to end
def run(*args):
    return cli.main([str(a) for a in args])


def test_parser_has_all_documented_commands():
    names = set(cli.build_parser()._subparsers._group_actions[0].choices)
    for c in ("env", "registry", "download-figshare", "import-figshare", "import-folder", "audit-manifest", "dedup", "split", "train", "seeds", "cv",
              "predict", "evaluate", "compare", "gradcam", "robustness", "efficiency", "audit-manuscript", "audit-parameters", "report-tables",
              "figures", "rag-demo-collection", "rag-build", "rag-generate", "rag-eval", "claims"):
        assert c in names, c


def test_cli_registry_and_env(capsys):
    run("registry", "--source", "zenodo_7786009_model_record")
    assert json.loads(capsys.readouterr().out)["contains_images"] is False
    run("env")
    assert "packages" in json.loads(capsys.readouterr().out)


def test_cli_data_pipeline(tmp_path, synth_root, capsys):
    man = tmp_path / "m.csv"
    run("import-folder", "--root", synth_root, "--source-id", "syn", "--manifest", man)
    assert "NOT established" in capsys.readouterr().out
    run("split", "--manifest", man, "--mode", "fixed", "--out", tmp_path / "sp", "--filename-pattern", r"(pat\d+)_s\d", "--seed", 1)
    rep = json.loads(capsys.readouterr().out)
    assert rep["groups_spanning_multiple_splits"] == 0
    run("audit-manifest", "--manifest", tmp_path / "sp" / "manifest_fixed.csv", "--expected")
    assert "difference_from_expected" in json.loads(capsys.readouterr().out)["split_audit"]
    run("dedup", "--manifest", man, "--root", synth_root, "--out", tmp_path / "dd")
    assert (tmp_path / "dd" / "perceptual_candidates.csv").is_file()
    run("split", "--manifest", man, "--mode", "cv", "--out", tmp_path / "cv", "--k", 3, "--filename-pattern", r"(pat\d+)_s\d")
    assert len(list((tmp_path / "cv").glob("manifest_fold*.csv"))) == 3


def test_cli_audit_manuscript_and_params(tmp_path, capsys):
    run("audit-manuscript", "--out", tmp_path / "a.md")
    assert (tmp_path / "a.md").read_text().startswith("# Numerical audit")
    run("audit-parameters", "--out", tmp_path / "p.json")
    assert "differences_M" in json.loads((tmp_path / "p.json").read_text())


def test_cli_eval_compare_tables(tmp_path, capsys):
    from conftest import synthetic_predictions
    from neurofusion.evaluation.predict import save_predictions

    a = save_predictions(synthetic_predictions(120, acc=0.95, seed=1), tmp_path / "a.csv")
    b = save_predictions(synthetic_predictions(120, acc=0.8, seed=1), tmp_path / "b.csv")  # same cases and labels, different model
    run("evaluate", "--predictions", a)
    assert "classification" in json.loads(capsys.readouterr().out)
    run("compare", "--reference", a, "--others", b, "--n-boot", 20)
    assert "p_exact" in capsys.readouterr().out
    run("report-tables", "--predictions", f"A={a}", f"B={b}", "--out", tmp_path / "t", "--n-boot", 20)
    assert (tmp_path / "t" / "results_overall.md").is_file()
    run("figures", "--predictions", a, "--out", tmp_path / "f")
    assert (tmp_path / "f" / "confusion_matrix.png").is_file()


def test_cli_rag_flow(tmp_path, capsys):
    run("rag-demo-collection", "--out", tmp_path / "demo")
    capsys.readouterr()
    run("rag-build", "--sources", tmp_path / "demo" / "source_manifest.csv", "--out", tmp_path / "idx", "--allow-demo")
    rep = json.loads(capsys.readouterr().out)
    assert rep["source_documents"] == 8 and rep["chunks"] == 8
    run("rag-generate", "--index", tmp_path / "idx", "--category", "meningioma", "--confidence", 0.9)
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok" and out["output"]["predicted_category"] == "meningioma"
    run("rag-build", "--sources", tmp_path / "demo" / "source_manifest.csv", "--out", tmp_path / "idx2", "--store", "qdrant")
    assert json.loads(capsys.readouterr().out)["excluded_unreviewed"] == 8  # demo docs are not indexed unless allowed
    (tmp_path / "run.json").write_text(json.dumps({"runs": {"q": ["a", "b"]}, "qrels": {"q": ["b"]}}))
    run("rag-eval", "--run", tmp_path / "run.json")
    assert json.loads(capsys.readouterr().out)["mrr@10"] == 0.5


def test_cli_rag_build_fails_on_tampered_sources(tmp_path, capsys):
    run("rag-demo-collection", "--out", tmp_path / "demo")
    (tmp_path / "demo" / "docs" / "demo-glioma-01.txt").write_text("tampered")
    with pytest.raises(SystemExit, match="verification failed"):
        run("rag-build", "--sources", tmp_path / "demo" / "source_manifest.csv", "--allow-demo")


def test_cli_train_predict_gradcam_robustness_smoke(tmp_path, synth_root, capsys, monkeypatch):
    """Whole classification CLI path on SYNTHETIC data with a tiny stand-in model (software smoke test only)."""
    import torch

    from conftest import TinyBranch, TinyViT
    from neurofusion.models.classifier import FusionClassifier, ModelConfig

    def builder(name, pretrained):
        return lambda: FusionClassifier(ModelConfig(pretrained=False, hidden1=16, hidden2=8), TinyBranch(24), TinyViT(16), 24, 16)

    monkeypatch.setattr(cli, "_model_builder", builder)
    man = tmp_path / "m.csv"
    run("import-folder", "--root", synth_root, "--source-id", "syn", "--manifest", man)
    run("split", "--manifest", man, "--mode", "fixed", "--out", tmp_path / "sp", "--filename-pattern", r"(pat\d+)_s\d")
    capsys.readouterr()
    fixed = tmp_path / "sp" / "manifest_fixed.csv"
    run("train", "--manifest", fixed, "--root", synth_root, "--out", tmp_path / "run", "--max-epochs", 2, "--device", "cpu", "--batch-size", 8)
    out = json.loads(capsys.readouterr().out)
    assert out["train_summary"]["epochs_run"] == 2 and "NEW EXPERIMENT" in out["status"]
    ck = tmp_path / "run" / "best.pt"
    run("predict", "--checkpoint", ck, "--manifest", fixed, "--root", synth_root, "--out", tmp_path / "pred.csv", "--device", "cpu")
    assert (tmp_path / "pred.csv").is_file()
    run("robustness", "--checkpoint", ck, "--manifest", fixed, "--root", synth_root, "--out", tmp_path / "rob.csv", "--device", "cpu")
    assert len(pd.read_csv(tmp_path / "rob.csv")) == 36
    img = next(synth_root.rglob("*.png"))
    for branch in ("a", "b"):
        run("gradcam", "--checkpoint", ck, "--image", img, "--out", tmp_path / f"cam{branch}", "--branch", branch)
        assert (tmp_path / f"cam{branch}" / f"{img.stem}.cam_raw.npy").is_file()
    run("efficiency", "--iters", 2, "--device", "cpu", "--out", tmp_path / "eff.json")
    assert torch is not None and (tmp_path / "eff.json").is_file()
