import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_predictions
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from neurofusion.constants import CLASS_NAMES
from neurofusion.evaluation import calibration as C
from neurofusion.evaluation.bootstrap import bootstrap_ci, paired_bootstrap_difference
from neurofusion.evaluation.metrics import accuracy, classification_report, macro_auc
from neurofusion.evaluation.paired import (
    AlignmentError,
    align_predictions,
    bonferroni,
    compare_to_reference,
    holm,
    mcnemar,
)
from neurofusion.evaluation.predict import PredictionError, load_predictions, save_predictions


def test_report_matches_sklearn():
    df = synthetic_predictions(400, acc=0.85, seed=1)
    r = classification_report(df)
    y, p = df["y_true"], df["y_pred"]
    assert r["accuracy"] == pytest.approx(accuracy_score(y, p)) and r["correct"] == int((y == p).sum())
    assert r["macro"]["precision"] == pytest.approx(precision_score(y, p, average="macro"))
    assert r["macro"]["recall"] == pytest.approx(recall_score(y, p, average="macro"))
    assert r["macro"]["f1"] == pytest.approx(f1_score(y, p, average="macro"))
    P = df[[f"p_{c}" for c in CLASS_NAMES]].to_numpy()
    assert r["macro"]["auc"] == pytest.approx(roc_auc_score(y, P, multi_class="ovr", average="macro"))
    assert np.array(r["confusion_matrix"]).sum() == 400


def test_macro_recall_equals_accuracy_when_balanced():
    n = 100
    y = np.repeat(np.arange(4), n)
    rng = np.random.default_rng(0)
    pred = np.where(rng.random(4 * n) < 0.9, y, (y + 1) % 4)
    P = np.eye(4)[pred] * 0.9 + 0.025
    df = pd.DataFrame(P, columns=[f"p_{c}" for c in CLASS_NAMES])
    df.insert(0, "y_pred", pred)
    df.insert(0, "y_true", y)
    df.insert(0, "image_id", range(4 * n))
    r = classification_report(df)
    assert r["macro"]["recall"] == pytest.approx(r["accuracy"])


def test_undefined_metrics_reported_not_hidden():
    df = synthetic_predictions(120, acc=0.9, seed=2)
    df = df[df["y_true"] != 2].copy()  # class 2 absent from ground truth
    r = classification_report(df)
    assert "notumor" in r["classes_absent"]
    assert "notumor:recall" in r["undefined"] and "notumor:auc" in r["undefined"]
    assert r["macro"]["recall_n_classes"] == 3
    assert np.isfinite(r["macro"]["recall"])


def test_ovr_mcc_is_not_multiclass_mcc():
    r = classification_report(synthetic_predictions(300, acc=0.8, seed=3))
    assert r["multiclass_mcc"] != pytest.approx(r["macro"]["ovr_mcc"])


def test_calibration_known_cases():
    perfect = synthetic_predictions(50, acc=1.0, seed=4)
    P = np.eye(4)[perfect["y_true"]]
    for i, c in enumerate(CLASS_NAMES):
        perfect[f"p_{c}"] = P[:, i]
    perfect["y_pred"] = perfect["y_true"]
    assert C.ece_mce(perfect)["ece"] == pytest.approx(0, abs=1e-9)
    assert C.brier(perfect) == pytest.approx(0) and C.nll(perfect) == pytest.approx(0, abs=1e-9)
    assert C.aurc(perfect) == 0 and C.coverage_at_risk(perfect, 0.01) == 1.0
    # uniform predictions, always predicting class 0
    u = perfect.copy()
    for c in CLASS_NAMES:
        u[f"p_{c}"] = 0.25
    assert C.brier(u) == pytest.approx(0.75) and C.nll(u) == pytest.approx(np.log(4))


def test_risk_coverage_monotone_coverage_and_report_keys():
    df = synthetic_predictions(200, acc=0.8, seed=5)
    rc = C.risk_coverage(df)
    assert rc["coverage"].is_monotonic_increasing and rc["coverage"].iloc[-1] == 1.0
    assert rc["risk"].iloc[-1] == pytest.approx(1 - accuracy(df))
    assert {"ece", "mce", "brier", "nll", "aurc", "coverage_at_risk_1pct", "mean_entropy"} <= set(C.calibration_report(df))


def test_bootstrap_deterministic_and_covers_point():
    df = synthetic_predictions(300, acc=0.9, seed=6)
    a = bootstrap_ci(df, accuracy, n_boot=300, seed=1)
    b = bootstrap_ci(df, accuracy, n_boot=300, seed=1)
    assert a == b and a["lo"] <= a["point"] <= a["hi"]
    assert "independent" in a["unit"]


def test_cluster_bootstrap_is_wider_when_clustered():
    n, g = 400, 20
    rng = np.random.default_rng(0)
    cluster = np.repeat(np.arange(g), n // g)
    eff = rng.random(g) < 0.5  # whole clusters right or wrong -> strong dependence
    df = synthetic_predictions(n, acc=0.5, seed=7)
    df["y_pred"] = np.where(eff[cluster], df["y_true"], (df["y_true"] + 1) % 4)
    df["cl"] = cluster
    img = bootstrap_ci(df, accuracy, n_boot=400, seed=0)
    clu = bootstrap_ci(df, accuracy, n_boot=400, seed=0, cluster_col="cl")
    assert (clu["hi"] - clu["lo"]) > (img["hi"] - img["lo"])
    assert clu["unit"].startswith("cluster")


def test_paired_bootstrap_shares_resampling_and_detects_difference():
    a = synthetic_predictions(300, acc=0.97, seed=8)
    b = a.copy()
    flip = np.arange(300) % 5 == 0
    b.loc[flip, "y_pred"] = (b.loc[flip, "y_true"] + 1) % 4
    r = paired_bootstrap_difference(a, b, accuracy, n_boot=300, seed=0)
    assert r["difference"] > 0 and r["lo"] > 0 and r["p_bootstrap_two_sided"] < 0.05
    same = paired_bootstrap_difference(a, a, accuracy, n_boot=50, seed=0)
    assert same["difference"] == 0 and same["lo"] == 0 == same["hi"]
    assert 0 <= paired_bootstrap_difference(a, b, macro_auc, n_boot=30, seed=0)["p_bootstrap_two_sided"] <= 1


def test_alignment_checks():
    a = synthetic_predictions(50, seed=9)
    with pytest.raises(AlignmentError, match="image_id"):
        align_predictions(a, a.iloc[:-1])
    b = a.copy()
    b.loc[0, "y_true"] = (b.loc[0, "y_true"] + 1) % 4
    with pytest.raises(AlignmentError, match="ground-truth"):
        align_predictions(a, b)
    shuffled = a.sample(frac=1, random_state=0)
    x, y = align_predictions(a, shuffled)
    assert x["image_id"].tolist() == y["image_id"].tolist()


def test_mcnemar_known_counts():
    n = 200
    a = synthetic_predictions(n, acc=1.0, seed=10)
    a["y_pred"] = a["y_true"]
    b = a.copy()
    wrong_idx = np.arange(7)
    b.loc[wrong_idx, "y_pred"] = (b.loc[wrong_idx, "y_true"] + 1) % 4  # A right, B wrong on 7 cases; none the other way
    r = mcnemar(a, b)
    assert (r["n10_a_right_b_wrong"], r["n01_a_wrong_b_right"]) == (7, 0)
    assert r["chi2_uncorrected"] == pytest.approx(7.0) and r["p_chi2_uncorrected"] == pytest.approx(stats.chi2.sf(7, 1))
    assert r["chi2_corrected"] == pytest.approx(36 / 7)
    assert r["p_exact"] == pytest.approx(2 * 0.5**7) and r["p_exact"] > r["p_chi2_uncorrected"]  # exact is more conservative
    assert r["recommended"] == "p_exact"


def test_mcnemar_not_derived_from_aggregate_accuracy():
    a = synthetic_predictions(100, acc=0.9, seed=11)
    b = a.copy()
    b["y_pred"] = a["y_pred"].to_numpy()[::-1]
    b["y_true"] = a["y_true"]
    with pytest.raises(AlignmentError):  # same aggregate accuracy construction but different cases must be caught by the join
        mcnemar(a, b.assign(image_id=b["image_id"] + "x"))


def test_mcnemar_no_discordance():
    a = synthetic_predictions(40, seed=12)
    r = mcnemar(a, a)
    assert r["discordant"] == 0 and r["p_exact"] == 1.0


def test_multiplicity_corrections():
    p = [0.01, 0.04, 0.03]
    assert holm(p) == pytest.approx([0.03, 0.06, 0.06])
    assert bonferroni(p) == pytest.approx([0.03, 0.12, 0.09])


def test_compare_to_reference_declares_family():
    ref = synthetic_predictions(120, acc=0.95, seed=13)
    others = {f"m{i}": ref.assign(y_pred=np.where(np.arange(120) % (i + 4) == 0, (ref["y_true"] + 1) % 4, ref["y_true"])) for i in range(3)}
    t = compare_to_reference(ref, others, "family-A")
    assert (t["family_size"] == 3).all() and (t["family"] == "family-A").all()
    assert (t["p_exact_holm"] >= t["p_exact"] - 1e-12).all()


def test_prediction_file_validation(tmp_path):
    df = synthetic_predictions(30, seed=14)
    p = save_predictions(df, tmp_path / "p.csv", {"m": 1})
    assert load_predictions(p).shape[0] == 30 and p.with_suffix(".meta.json").is_file()
    bad = df.copy()
    bad.loc[0, "p_glioma"] += 0.5
    save_predictions(bad, tmp_path / "bad.csv")
    with pytest.raises(PredictionError, match="sum to 1"):
        load_predictions(tmp_path / "bad.csv")
    inc = df.copy()
    inc["y_pred"] = (inc["y_pred"] + 1) % 4
    save_predictions(inc, tmp_path / "inc.csv")
    with pytest.raises(PredictionError, match="argmax"):
        load_predictions(tmp_path / "inc.csv")
    save_predictions(df.drop(columns="p_notumor"), tmp_path / "miss.csv")
    with pytest.raises(PredictionError, match="missing"):
        load_predictions(tmp_path / "miss.csv")
