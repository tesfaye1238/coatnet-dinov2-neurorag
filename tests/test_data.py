import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from neurofusion.constants import CLASS_NAMES, MANUSCRIPT_TABLE1
from neurofusion.data import splits as S
from neurofusion.data.dataset import ManifestDataset
from neurofusion.data.dedup import (
    apply_pair_decisions,
    exact_duplicate_groups,
    perceptual_candidate_pairs,
    resolve_exact_duplicates,
)
from neurofusion.data.grouping import assign_groups, merge_groups_by_pairs
from neurofusion.data.labels import LabelError, normalize_label
from neurofusion.data.manifest import (
    ManifestError,
    build_manifest_from_folder,
    load_manifest,
    save_manifest,
    validate_manifest,
)
from neurofusion.data.transforms import AugmentationConfig, build_eval_transform, build_train_transform


# ---- labels / manifest
def test_label_aliases():
    assert normalize_label("No Tumor") == "notumor"
    assert normalize_label("pituitary_tumor") == "pituitary"


@pytest.mark.parametrize("raw", ["yes", "no", "tumor", "normal"])
def test_binary_labels_rejected(raw):
    with pytest.raises(LabelError, match="binary"):
        normalize_label(raw)


def test_unknown_label_rejected():
    with pytest.raises(LabelError):
        normalize_label("astrocytoma")


def test_manifest_built_without_inventing_patients(synth_root):
    df = build_manifest_from_folder(synth_root, "src", "r1")
    assert len(df) == 48
    assert (df["patient_id"] == "").all()
    assert not df["patient_id_verified"].any()
    assert set(df["label"]) == set(CLASS_NAMES)
    assert df["file_sha256"].str.len().eq(64).all()
    validate_manifest(df)


def test_manifest_requires_patient_id_when_verified(synth_manifest):
    bad = synth_manifest.copy()
    bad.loc[0, "patient_id_verified"] = True
    with pytest.raises(ManifestError, match="invented"):
        validate_manifest(bad)


def test_manifest_rejects_duplicate_ids_and_bad_labels(synth_manifest):
    dup = pd.concat([synth_manifest, synth_manifest.iloc[:1]])
    with pytest.raises(ManifestError, match="Duplicate"):
        validate_manifest(dup)
    bad = synth_manifest.copy()
    bad.loc[0, "label"] = "yes"
    with pytest.raises(ManifestError, match="Labels"):
        validate_manifest(bad)


def test_manifest_detects_group_spanning_splits(synth_manifest):
    m = synth_manifest.copy()
    g = m["group_id"].iloc[0]
    idx = m.index[m["group_id"] == g]
    m["split"] = "train"
    m.loc[idx[0], "split"] = "test"
    with pytest.raises(ManifestError, match="span more than one split"):
        validate_manifest(m)


def test_manifest_roundtrip(tmp_path, synth_manifest):
    p = save_manifest(synth_manifest, tmp_path / "m.csv")
    back = load_manifest(p)
    assert back["image_id"].tolist() == synth_manifest["image_id"].tolist()
    assert back["patient_id_verified"].dtype == bool


def test_empty_folder_raises(tmp_path):
    (tmp_path / "x").mkdir()
    with pytest.raises(ManifestError):
        build_manifest_from_folder(tmp_path / "x", "s")


# ---- dedup
def test_exact_and_decoded_duplicates(tmp_path):
    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, (32, 32), dtype=np.uint8)
    for name in ("a.png", "b.png"):
        (tmp_path / "glioma").mkdir(exist_ok=True)
        Image.fromarray(a).save(tmp_path / "glioma" / name)
    Image.fromarray(a).convert("RGB").save(tmp_path / "glioma" / "c_rgb.png")  # same pixels once decoded to RGB
    Image.fromarray(rng.integers(0, 255, (32, 32), dtype=np.uint8)).save(tmp_path / "glioma" / "d.png")
    df = build_manifest_from_folder(tmp_path, "s")
    ex = exact_duplicate_groups(df)
    assert set(ex["image_id"]) == {"s/glioma/a.png", "s/glioma/b.png"}
    dec = exact_duplicate_groups(df, "decoded_image_hash")
    assert len(dec) == 3  # a, b and the RGB re-encode share decoded pixels
    kept, log = resolve_exact_duplicates(df, "decoded_image_hash")
    assert len(kept) == 2 and len(log) == 2
    assert (log["retained_image_id"] == "s/glioma/a.png").all()
    assert (tmp_path / "glioma" / "b.png").exists()  # nothing deleted on disk


def test_perceptual_candidates_are_flagged_not_removed(tmp_path):
    rng = np.random.default_rng(2)
    base = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    noisy = np.clip(base.astype(int) + rng.integers(-3, 4, base.shape), 0, 255).astype(np.uint8)
    other = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    (tmp_path / "meningioma").mkdir()
    for n, a in (("a", base), ("b", noisy), ("c", other)):
        Image.fromarray(a).save(tmp_path / "meningioma" / f"{n}.png")
    df = build_manifest_from_folder(tmp_path, "s")
    pairs = perceptual_candidate_pairs(df, tmp_path, "dhash", max_hamming=8)
    assert {(r.image_id_a, r.image_id_b) for r in pairs.itertuples()} == {("s/meningioma/a.png", "s/meningioma/b.png")}
    assert (pairs["decision"] == "pending").all()
    kept, _ = apply_pair_decisions(df, pairs)  # pending keeps both
    assert len(kept) == 3
    pairs["decision"] = "keep_a"
    kept, log = apply_pair_decisions(df, pairs)
    assert len(kept) == 2 and log["removed_image_id"].tolist() == ["s/meningioma/b.png"]


def test_label_conflict_flagged_in_pairs(tmp_path):
    rng = np.random.default_rng(3)
    a = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    for c in ("glioma", "notumor"):
        (tmp_path / c).mkdir()
        Image.fromarray(a).save(tmp_path / c / "x.png")
    df = build_manifest_from_folder(tmp_path, "s")
    pairs = perceptual_candidate_pairs(df, tmp_path, "phash", 2)
    assert bool(pairs.iloc[0]["label_conflict"])


# ---- grouping
def test_grouping_methods(synth_root):
    df = build_manifest_from_folder(synth_root, "s")
    df.loc[0, ["patient_id", "patient_id_verified"]] = ["P1", True]
    g = assign_groups(df, r"(pat\d+)_s\d")
    assert g.loc[0, "grouping_method"] == "verified_patient_id" and g.loc[0, "group_id"] == "patient:s:P1"
    assert (g.loc[1:, "grouping_method"] == "heuristic_filename").all()
    g2 = assign_groups(df)
    assert (g2.loc[1:, "grouping_method"] == "singleton").all()
    assert not g["patient_id_verified"].iloc[1:].any()  # heuristics never become verified identity


def test_merge_groups_by_pairs(synth_manifest):
    ids = synth_manifest["image_id"].tolist()
    a, b = ids[0], ids[5]
    pairs = pd.DataFrame([{"image_id_a": a, "image_id_b": b}])
    out = merge_groups_by_pairs(synth_manifest, pairs)
    ga, gb = out.set_index("image_id").loc[[a, b], "group_id"]
    assert ga == gb


# ---- splits
def _check_integrity(df):
    assert (df.groupby("group_id")["split"].nunique() == 1).all()


def test_group_split_never_splits_a_group(synth_manifest):
    out = S.group_stratified_split(synth_manifest, 0.17, 0.33, seed=1)
    _check_integrity(out)
    rep = S.audit_split(out)
    assert rep["groups_spanning_multiple_splits"] == 0
    assert set(out["split"]) == {"train", "val", "test"}
    assert rep["patient_level_independence_established"] is False  # heuristic groups are not verified identities


def test_split_is_deterministic_and_seed_dependent(synth_manifest):
    a = S.group_stratified_split(synth_manifest, seed=3)
    b = S.group_stratified_split(synth_manifest, seed=3)
    c = S.group_stratified_split(synth_manifest, seed=4)
    assert a["split"].tolist() == b["split"].tolist()
    assert a["split"].tolist() != c["split"].tolist()


def test_mixed_label_group_rejected(synth_manifest):
    m = synth_manifest.copy()
    m.loc[m.index[:2], "group_id"] = "same"
    m.loc[m.index[0], "label"] = "glioma"
    m.loc[m.index[1], "label"] = "notumor"
    with pytest.raises(S.SplitError, match="more than one label"):
        S.group_stratified_split(m)


def test_kfold_tests_partition_data_and_val_from_pool(synth_manifest):
    folds = S.kfold_splits(synth_manifest, k=3, seed=0)
    tests = [set(f.loc[f["split"] == "test", "image_id"]) for f in folds]
    assert set.union(*tests) == set(synth_manifest["image_id"])
    assert sum(len(t) for t in tests) == len(synth_manifest)  # disjoint test folds
    for f in folds:
        _check_integrity(f)
        assert (f["split"] == "val").sum() > 0
        assert not set(f.loc[f["split"] == "val", "image_id"]) & set(f.loc[f["split"] == "test", "image_id"])


def test_source_held_out_checks_class_availability(synth_manifest):
    m = synth_manifest.copy()
    half = len(m) // 2
    m.loc[m.index[:half], "source_id"] = "A"
    m.loc[m.index[half:], "source_id"] = "B"
    m = m[~((m["source_id"] == "A") & (m["label"] == "notumor"))].copy()  # A lacks notumor
    rep = S.check_source_compatibility(m, "A")
    assert "notumor" in rep["classes_absent_in_test"]
    with pytest.raises(S.SplitError, match="occur in test source"):
        S.source_held_out_split(m[~((m["source_id"] == "B") & (m["label"] == "notumor"))], "A")  # fine: A has none; error only when train lacks a class A has
    # train (B) lacks glioma while test (A) has it
    m2 = m[~((m["source_id"] == "B") & (m["label"] == "glioma"))]
    with pytest.raises(S.SplitError):
        S.source_held_out_split(m2, "A")


def test_source_held_out_ok(synth_manifest):
    m = synth_manifest.copy()
    m["source_id"] = ["A" if i % 2 else "B" for i in range(len(m))]
    out = S.source_held_out_split(m, "A", seed=0)
    assert set(out.loc[out["source_id"] == "A", "split"]) == {"test"}
    assert "test" not in set(out.loc[out["source_id"] == "B", "split"])


def test_import_split_requires_complete_file(tmp_path, synth_manifest):
    sp = pd.DataFrame({"image_id": synth_manifest["image_id"][:-3], "split": "train"})
    sp.to_csv(tmp_path / "s.csv", index=False)
    with pytest.raises(S.SplitError, match="no split"):
        S.import_split_manifest(synth_manifest, tmp_path / "s.csv")


def test_audit_reports_differences_from_expected_without_forcing(synth_manifest):
    out = S.group_stratified_split(synth_manifest, seed=0)
    rep = S.audit_split(out, MANUSCRIPT_TABLE1)
    d = rep["difference_from_expected"]["glioma/test"]
    assert d["expected"] == 400 and d["actual"] < 400 and d["diff"] == d["actual"] - 400


def test_verified_patient_independence_detected(synth_root):
    df = build_manifest_from_folder(synth_root, "s", patient_id_fn=lambda p: p.stem.split("_")[0], patient_ids_verified=True)
    df = assign_groups(df)
    out = S.group_stratified_split(df, seed=0)
    assert S.audit_split(out)["patient_level_independence_established"] is True


# ---- transforms / dataset
def test_eval_transform_deterministic_and_normalised(synth_root):
    img = Image.open(next(synth_root.rglob("*.png")))
    t = build_eval_transform()
    a, b = t(img), t(img)
    assert torch.equal(a, b) and a.shape == (3, 224, 224)
    raw = np.asarray(img.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255
    assert abs(float(a[0].mean()) - float(((raw[..., 0] - 0.485) / 0.229).mean())) < 0.1


def test_train_transform_is_stochastic_but_eval_is_not(synth_root):
    img = Image.open(next(synth_root.rglob("*.png")))
    tt = build_train_transform()
    outs = [tt(img) for _ in range(6)]
    assert any(not torch.equal(outs[0], o) for o in outs[1:])


def test_vertical_flip_sensitivity_config():
    cfg = AugmentationConfig(vflip_p=0.0)
    assert cfg.vflip_p == 0.0 and AugmentationConfig().vflip_p == 0.5
    assert AugmentationConfig().noise_std == 0.01 and AugmentationConfig().rotation_deg == 15


def test_dataset_missing_files(tmp_path, synth_manifest):
    with pytest.raises(FileNotFoundError):
        ManifestDataset(synth_manifest, tmp_path / "nowhere")


def test_dataset_item(synth_root, synth_manifest):
    ds = ManifestDataset(synth_manifest, synth_root, build_eval_transform())
    x, y, iid = ds[0]
    assert x.shape == (3, 224, 224) and y == CLASS_NAMES.index(synth_manifest.loc[0, "label"]) and iid == synth_manifest.loc[0, "image_id"]
