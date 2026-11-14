import pandas as pd
import pytest
from conftest import synthetic_predictions

from neurofusion.evaluation.predict import save_predictions
from neurofusion.reporting.numeric_audit import (
    attainable,
    clopper_pearson,
    cm_metrics,
    infer_discordant,
    load_reference,
    render_markdown,
    run_audit,
    wilson,
)
from neurofusion.reporting.parameter_audit import parameter_audit
from neurofusion.reporting.reconcile import reconcile_accuracy
from neurofusion.reporting.tables import (
    per_class_table,
    plot_confusion,
    results_table,
    save_table,
    to_markdown,
)


def test_attainability():
    ok, k, v = attainable(99.31, 1600)
    assert ok and k == 1589
    ok, k, v = attainable(99.08, 1600)
    assert not ok and k == 1585
    assert attainable(98.43, 1405)[0] in (True, False)


def test_wilson_and_cp_contain_point():
    lo, hi = wilson(1589, 1600)
    assert lo < 1589 / 1600 < hi
    lo, hi = clopper_pearson(1589, 1600)
    assert lo < 1589 / 1600 < hi and clopper_pearson(0, 10)[0] == 0.0 and clopper_pearson(10, 10)[1] == 1.0


def test_infer_discordant_reproduces_mcnemar_table():
    assert (7, 0, 7.0) in infer_discordant(7.0, 7.2)
    assert any(b == 17 and c == 1 for b, c, _ in infer_discordant(14.22, 16.0))
    assert infer_discordant(123.456, 3.0) == []


def test_cm_metrics_known():
    m = cm_metrics([[9, 1], [1, 9]])
    assert m["acc"] == 0.9 and m["macro_f1"] == pytest.approx(0.9)


def test_full_audit_runs_and_flags_known_inconsistencies():
    checks = {c.id: c for c in run_audit()}
    assert checks["A01"].status == "PASS"
    assert checks["A02"].status == "FAIL" and "5712" in checks["A02"].detail
    assert checks["A03"].status == "FAIL" and "GhostNetV2" in checks["A03"].detail
    assert checks["A06"].status == "PASS" and checks["A07"].status == "PASS"
    assert checks["A08"].status == "FAIL"
    assert checks["A10"].status == "PASS"
    assert checks["A13"].status == "FAIL" and checks["A16"].status == "FAIL"
    assert "exact p" in checks["A15"].detail
    assert "15.3%" in checks["A19"].detail and "8.5%" in checks["A19"].detail
    md = render_markdown(list(checks.values()))
    assert md.startswith("# Numerical audit") and "A15" in md


def test_reference_file_has_provenance():
    ref = load_reference()
    assert "PAPER-REPORTED" in ref["_provenance"] and ref["n_test"] == 1600


def test_parameter_audit_supports_differences():
    rep = parameter_audit()
    d = rep["differences_M"]
    for k, v in d.items():
        assert abs(v["instantiated"] - v["reported"]) < 0.01, k
    offs = [v["offset_M"] for v in rep["variants"].values()]
    assert max(offs) - min(offs) < 0.01  # constant offset, i.e. a backbone-level difference


def test_results_tables_from_saved_predictions(tmp_path):
    p1 = save_predictions(synthetic_predictions(150, acc=0.9, seed=1), tmp_path / "a.csv")
    p2 = save_predictions(synthetic_predictions(150, acc=0.8, seed=2), tmp_path / "b.csv")
    t = results_table({"A": p1, "B": p2}, n_boot=50)
    assert set(t["model"]) == {"A", "B"} and (t["acc_ci_lo_%"] <= t["accuracy_%"]).all()
    assert "independent" in t["interval_unit"].iloc[0]
    save_table(t, tmp_path / "out", "r")
    assert (tmp_path / "out" / "r.md").read_text().startswith("| model")
    assert len(per_class_table(p1)) == 4
    assert plot_confusion(p1, tmp_path / "cm.png").is_file()
    assert isinstance(to_markdown(pd.DataFrame({"a": [1.5], "b": ["x"]})), str)


def test_reconcile_marks_new_experiment(tmp_path):
    p = save_predictions(synthetic_predictions(100, acc=0.95, seed=3), tmp_path / "p.csv")
    r = reconcile_accuracy({"proposed": p, "mystery": p})
    row = r.set_index("model")
    assert "diff_acc_pp" in row.loc["proposed"].dropna().index
    assert row.loc["mystery", "note"].startswith("no manuscript")
    assert "NEW EXPERIMENT" in row["status"].iloc[0]
