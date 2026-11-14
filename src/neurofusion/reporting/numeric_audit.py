"""Numerical audit of the manuscript's own reported numbers.

Everything here is arithmetic on PAPER-REPORTED values (``reference/manuscript_reported.json``). It measures
internal consistency; it does not and cannot establish what the original experiments produced. Where a check
infers something (e.g. discordant counts from a McNemar chi-square) the inference and its assumption are stated.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

REFERENCE_PATH = Path(__file__).resolve().parents[3] / "reference" / "manuscript_reported.json"


@dataclass
class Check:
    id: str
    status: str  # PASS | FAIL | INFO
    title: str
    detail: str


def load_reference(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or REFERENCE_PATH).read_text(encoding="utf-8"))


def attainable(value_pct: float, n: int, dp: int = 2) -> tuple[bool, int, float]:
    """Is there an integer k with round(k/n*100, dp) == value_pct? Returns (ok, nearest_k, value_at_k)."""
    ks = np.arange(0, n + 1)
    vals = ks / n * 100
    tol = 0.5 * 10 ** (-dp) + 1e-9
    ok = np.abs(vals - value_pct) <= tol
    nearest = int(ks[np.argmin(np.abs(vals - value_pct))])
    return bool(ok.any()), nearest, float(vals[nearest])


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def cm_metrics(cm: np.ndarray) -> dict:
    cm = np.asarray(cm)
    n = cm.sum()
    tp = np.diag(cm).astype(float)
    prec = tp / cm.sum(0)
    rec = tp / cm.sum(1)
    f1 = 2 * prec * rec / (prec + rec)
    return {"n": int(n), "correct": int(tp.sum()), "acc": tp.sum() / n, "prec": prec, "rec": rec, "f1": f1,
            "macro_prec": prec.mean(), "macro_rec": rec.mean(), "macro_f1": f1.mean()}


def infer_discordant(chi2: float, delta_images: float, max_total: int = 200, slack: float = 1.5) -> list[tuple[int, int, float]]:
    """Integer (b, c) with (b-c)^2/(b+c) == chi2 (to 0.006) and |b-c| within ``slack`` images of ``delta_images``.
    b = proposed right & baseline wrong; c = the reverse. This is an INFERENCE assuming the uncorrected statistic."""
    out = []
    for m in range(1, max_total + 1):
        for b in range(m + 1):
            c = m - b
            if abs((b - c) ** 2 / m - chi2) <= 0.006 and abs((b - c) - delta_images) <= slack:
                out.append((b, c, (b - c) ** 2 / m))
    return out


def run_audit(ref: dict | None = None) -> list[Check]:
    ref = ref or load_reference()
    checks: list[Check] = []
    add = lambda *a: checks.append(Check(*a))  # noqa: E731
    n_test = ref["n_test"]

    # --- Table 1 arithmetic
    t1 = ref["table1"]
    bad = []
    for i, c in enumerate(t1["classes"]):
        if t1["raw"][i] - t1["exact"][i] - t1["near"][i] != t1["final"][i]:
            bad.append(f"{c}: raw-exact-near")
        if t1["train"][i] + t1["val"][i] + t1["test"][i] != t1["final"][i]:
            bad.append(f"{c}: train+val+test")
    tot_ok = all(sum(t1[k]) == v for k, v in t1["totals"].items())
    add("A01", "PASS" if not bad and tot_ok else "FAIL", "Table 1 row and column arithmetic",
        "All rows and totals reconcile (raw - exact - near = final; train + val + test = final)." if not bad and tot_ok else f"Problems: {bad}")

    # --- Figure 2 vs Table 1
    f2 = ref["figure2_class_counts"]
    f2_train = sum(f2["train"].values())
    add("A02", "FAIL" if f2_train != t1["totals"]["train"] else "PASS",
        "Figure 2 training counts vs Table 1 training split",
        f"Figure 2 pie labels sum to {f2_train} training images ({f2['train']}), Table 1 states {t1['totals']['train']} "
        f"(per class {dict(zip(t1['classes'], t1['train']))}). Test pie = 400/class matches Table 1, the training pie does not.")

    # --- accuracy attainability on n=1600
    rows = []
    for name, vals in ref["table4"]["rows"].items():
        ok, k, v = attainable(vals[0], n_test)
        rows.append((f"Table4 {name}", vals[0], ok, k, v))
    for name, r in ref["table5"]["rows"].items():
        ok, k, v = attainable(r["acc"], n_test)
        rows.append((f"Table5 {name}", r["acc"], ok, k, v))
    bad = [(a, b, k, round(v, 4)) for a, b, ok, k, v in rows if not ok]
    add("A03", "FAIL" if bad else "PASS", f"Reported accuracies attainable as k/{n_test} (2 dp)",
        "All attainable." if not bad else "Not attainable by any integer k/1600: " + "; ".join(
            f"{a} reported {b} (nearest k={k} gives {v})" for a, b, k, v in bad))

    # --- fold accuracies
    bad = []
    for i, f in enumerate(ref["table7A"]["folds"]):
        ok, k, v = attainable(f["acc"], f["test"])
        if not ok:
            bad.append(f"fold {i + 1}: {f['acc']} on n={f['test']} (nearest k={k}: {v:.4f})")
    add("A04", "FAIL" if bad else "PASS", "Table 7A fold accuracies attainable on their fold sizes",
        "All fold accuracies are attainable." if not bad else "; ".join(bad))

    # --- confusion matrix recomputation
    cm = cm_metrics(np.array(ref["confusion_matrix_figure8"]["matrix"]))
    t4 = ref["table4"]["rows"]["Proposed"]
    add("A05", "INFO", "Metrics recomputed from the Figure 8 confusion matrix",
        f"n={cm['n']}, correct={cm['correct']}, accuracy={cm['acc'] * 100:.3f}%, macro-precision={cm['macro_prec'] * 100:.3f}%, "
        f"macro-recall={cm['macro_rec'] * 100:.3f}%, macro-F1={cm['macro_f1'] * 100:.3f}%. Table 4 reports {t4}. "
        f"Per-class precision {np.round(cm['prec'], 4).tolist()}, recall {np.round(cm['rec'], 4).tolist()}, F1 {np.round(cm['f1'], 4).tolist()}.")
    ok_t4 = all(abs(x - y) < 0.01 for x, y in zip([cm["acc"] * 100, cm["macro_prec"] * 100, cm["macro_rec"] * 100, cm["macro_f1"] * 100], t4))
    add("A06", "PASS" if ok_t4 else "FAIL", "Table 4 'Proposed' row vs Figure 8 confusion matrix",
        "Table 4 agrees with the confusion matrix to 0.01 pp." if ok_t4 else "Table 4 disagrees with the confusion matrix.")
    pc = ref["per_class_section6_1"]
    pc_bad = [c for i, c in enumerate(["glioma", "meningioma", "notumor", "pituitary"])
              if abs(cm["prec"][i] - pc[c]["precision"]) > 6e-5 or abs(cm["rec"][i] - pc[c]["recall"]) > 6e-5 or abs(cm["f1"][i] - pc[c]["f1"]) > 6e-5]
    add("A07", "PASS" if not pc_bad else "FAIL", "Per-class precision/recall/F1 (Section 6.1) vs Figure 8",
        "Per-class values agree with the confusion matrix." if not pc_bad else f"Disagreement for {pc_bad}")
    tx = ref["section6_1_text"]
    add("A08", "FAIL", "Section 6.1 narrative vs Table 4 and the confusion matrix",
        f"Narrative gives precision {tx['precision']}, recall {tx['recall']}, F1 {tx['f1']}; the confusion matrix gives "
        f"{cm['macro_prec'] * 100:.2f}, {cm['macro_rec'] * 100:.2f}, {cm['macro_f1'] * 100:.2f} (Table 4: 99.31 each). "
        "For a class-balanced single-label test set macro-recall must equal accuracy "
        f"({cm['macro_rec'] * 100:.3f} == {cm['acc'] * 100:.3f}); the narrative recall {tx['recall']} cannot be correct.")
    add("A09", "INFO", "Table 5 macro-precision for the proposed model",
        f"Table 5 and Supplementary S4 give 99.32 / 0.9932; the confusion matrix gives {cm['macro_prec'] * 100:.3f} (rounds to 99.31, as in Table 4).")

    # --- Table 7A summary
    accs = [f["acc"] for f in ref["table7A"]["folds"]]
    f1s = [f["f1"] for f in ref["table7A"]["folds"]]
    aucs = [f["auc"] for f in ref["table7A"]["folds"]]
    m_a, s_a = statistics.mean(accs), statistics.stdev(accs)
    m_f, s_f = statistics.mean(f1s), statistics.stdev(f1s)
    m_u, s_u = statistics.mean(aucs), statistics.stdev(aucs)
    rep = ref["table7A"]["mean_sd"]
    ok = (round(m_a, 2), round(s_a, 2)) == tuple(rep["acc"]) and (round(m_f, 2), round(s_f, 2)) == tuple(rep["f1"]) and (round(m_u, 3), round(s_u, 3)) == tuple(rep["auc"])
    add("A10", "PASS" if ok else "FAIL", "Table 7A mean +/- sample SD",
        f"Recomputed acc {m_a:.3f}+/-{s_a:.3f}, F1 {m_f:.3f}+/-{s_f:.3f}, AUC {m_u:.4f}+/-{s_u:.4f}; reported {rep}.")
    sizes_ok = all(f["train"] + f["val"] + f["test"] == t1["totals"]["final"] for f in ref["table7A"]["folds"])
    add("A11", "PASS" if sizes_ok else "FAIL", "Table 7A fold sizes sum to the curated benchmark (7,022)",
        "Each fold: train + val + test = 7,022. Fold manifests are still required to verify that counts came from real folds.")

    # --- Table 7B
    b7 = ref["table7B"]["rows"]
    tot = sum(r["n_test"] for r in b7)
    drops_ok = all(abs((99.31 - r["acc"]) - r["drop"]) < 0.006 for r in b7)
    add("A12", "PASS" if tot == 7022 and drops_ok else "FAIL", "Table 7B arithmetic",
        f"Source test sizes sum to {tot}; drops recompute against 99.31: {drops_ok}.")
    ps = ref["provenance_statements"]
    src = ps["source_sizes_table7B"]
    add("A13", "FAIL", "Table 7B source sizes vs dataset provenance",
        f"Table 7B uses source sizes figshare={src['figshare']}, SARTAJ={src['SARTAJ']}, Br35H={src['Br35H']} (sum {sum(src.values())} = the curated total). "
        f"(i) Table 1 reports {ps['notumor_total_table1']} no-tumor images after screening, while the Zenodo record description says the no-tumor class came from Br35H; "
        f"Br35H holds only {src['Br35H']} images here, which is fewer than {ps['notumor_total_table1']}. (ii) The figshare collection has no no-tumor class (3 tumor classes only), "
        "so a 4-class macro metric on a figshare test source has an absent class. (iii) 218 duplicates were removed (Table 1) yet the figshare and SARTAJ subset sizes equal the "
        "full original collection sizes (3,064; the SARTAJ size is as stated by the manuscript). (iv) Br35H carries only binary tumor labels, so 4-class labels for that source need documented annotations. "
        "Manifests with source ids are required to resolve this.")

    # --- Table 6
    t6 = ref["table6"]["rows"]
    bad = []
    for name, r in t6.items():
        if abs(r["acc"] / r["params"] - r["acc_per_m"]) > 0.006:
            bad.append(f"{name} acc/param")
        if abs(1000 / r["latency"] - r["throughput"]) > 0.06:
            bad.append(f"{name} throughput")
    add("A14", "INFO", "Table 6 derived columns",
        ("Accuracy/M-params and throughput (= 1000/latency, i.e. batch size 1) reproduce arithmetically." if not bad else f"Mismatch: {bad}")
        + " Throughput equal to 1000/latency indicates batch-1 timing. The energy column has no stated measurement method.")

    # --- McNemar reverse engineering
    s5 = ref["tableS5"]["rows"]
    n_proposed_correct = round(99.31 / 100 * n_test)
    lines, flips = [], []
    thr = ref["tableS5"]["bonferroni_threshold"]
    for name, r in s5.items():
        cand = infer_discordant(r["mcnemar_chi2"], r["delta_acc"] / 100 * n_test)
        if not cand:
            lines.append(f"{name}: no integer (b,c) reproduces chi2={r['mcnemar_chi2']} with a ~{r['delta_acc'] / 100 * n_test:.1f}-image difference")
            continue
        b, c, chi = min(cand, key=lambda t: abs((t[0] - t[1]) - r["delta_acc"] / 100 * n_test))
        base_correct = n_proposed_correct - (b - c)
        t4_acc = ref["table4"]["rows"][name][0]
        p_chi = float(stats.chi2.sf(chi, 1))
        p_exact = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
        lines.append(f"{name}: implied (b={b}, c={c}); implied baseline correct={base_correct} -> {base_correct / n_test * 100:.2f}% vs Table 4 {t4_acc}; "
                     f"p(chi2, uncorrected)={p_chi:.4f}, exact p={p_exact:.4f}")
        if (p_chi < thr) != (p_exact < thr):
            flips.append(name)
    add("A15", "INFO", "Table S5 McNemar statistics reverse-engineered to integer discordant counts",
        "ASSUMPTIONS: uncorrected McNemar chi2 = (b-c)^2/(b+c), the proposed model has 1,589 correct, one test set of 1,600. " + " | ".join(lines)
        + (f" | Significance at the Bonferroni threshold {thr} flips between the uncorrected chi-square and the exact test for: {flips}." if flips else ""))
    add("A16", "FAIL", "Table S5 DeLong z for a macro (multiclass) AUC",
        "The ordinary DeLong test is defined for a binary AUC. The manuscript gives one z and p for 'macro-AUC' without describing a multiclass extension; "
        "it cannot be reproduced or validated without the original procedure and probabilities. This repository uses a paired bootstrap for macro-AUC differences.")

    # --- CI check
    bad = []
    for name in ("proposed", "concat_no_attention", "concat_se_r16", "sum_se_r8", "coatnet_only", "dinov2_only"):
        r = ref["table5"]["rows"][name]
        ok, k, _ = attainable(r["acc"], n_test)
        w = [round(x * 100, 2) for x in wilson(k, n_test)]
        cp = [round(x * 100, 2) for x in clopper_pearson(k, n_test)]
        bad.append(f"{name} (k={k}): reported {r['ci']}, Wilson {w}, Clopper-Pearson {cp}")
    add("A17", "INFO", "Reported 95% accuracy CIs vs analytic intervals for k/1600",
        "The original CI procedure is not described. " + " | ".join(bad))

    # --- Table 5 deltas
    p = ref["table5"]["rows"]["proposed"]["acc"]
    bad = [f"{n}: reported {r['delta']}, recomputed {round(r['acc'] - p, 2)}" for n, r in ref["table5"]["rows"].items() if abs(round(r["acc"] - p, 2) - r["delta"]) > 0.0051]
    add("A18", "PASS" if not bad else "INFO", "Table 5 delta column vs differences of the printed accuracies",
        "Deltas agree to rounding." if not bad else "Differences (rounding of 1/1600 = 0.0625 pp steps): " + "; ".join(bad))

    # --- claims
    cl = ref["claims"]
    tot = cl["supported"] + cl["unverifiable"] + cl["contradicted"]
    add("A19", "INFO", "Claim-count arithmetic (counts supplied in the brief, not in the manuscript)",
        f"{cl['supported']} + {cl['unverifiable']} + {cl['contradicted']} = {tot}; contradicted fraction {cl['contradicted'] / tot * 100:.1f}%, "
        f"combined non-supported fraction {(cl['unverifiable'] + cl['contradicted']) / tot * 100:.1f}%. The 8.5% rate is the contradicted share only. "
        "The arithmetic does not verify how annotations were produced (annotators, protocol, agreement).")

    # --- manual document findings
    manual = [
        ("M01", "Zenodo 10.5281/zenodo.7786009 is a trained-model record (model.h5 + metrics JSON), not an image archive; the Data Availability statement cites it as the dataset. Its metrics file reports a held-out distribution of 324/329/400/351 (n=1404), not 400 per class."),
        ("M02", "Figures 5 and 6 are captioned 'training and test' curves; Section 5.1 says the validation subset monitored training and the test set was not used for model selection."),
        ("M03", "Two sections are numbered 6.6 (Ablation Study; Model Efficiency...). Figure 11 labels 'Meninigioma' (typo) and figure labels use 'Notumor'."),
        ("M04", "Ablation Table 5 contains a variant named 'Element-wise sum, SE r=8' whose projection is undescribed; parameter counts imply a 384->768 projection (see parameter audit)."),
        ("M05", "AP for the no-tumor class (0.9947) is lower than for glioma (0.9994) although no-tumor has the highest precision, recall and F1; AUC glioma 0.993 vs AP 0.9994 is also unusual. These need the saved probabilities to check."),
        ("M06", "Table 3 states the same training configuration was used for all baselines; Table 6 reports a 'Train / epoch' of 38-248 s without stating hardware, batch size or data-loading settings."),
    ]
    for i, t in manual:
        add(i, "INFO", "Manual document finding", t)
    return checks


def render_markdown(checks: list[Check]) -> str:
    counts = {s: sum(c.status == s for c in checks) for s in ("PASS", "FAIL", "INFO")}
    out = [
        "# Numerical audit of the manuscript's reported numbers",
        "",
        "Generated by `neurofusion audit-manuscript` from `reference/manuscript_reported.json` (paper-reported values).",
        "This checks internal consistency only. It does not establish what the original experiments produced.",
        "",
        f"PASS: {counts['PASS']}  FAIL: {counts['FAIL']}  INFO: {counts['INFO']}",
        "",
        "| ID | Status | Check |",
        "|---|---|---|",
    ]
    out += [f"| {c.id} | {c.status} | {c.title} |" for c in checks]
    out.append("")
    for c in checks:
        out += [f"## {c.id} [{c.status}] {c.title}", "", c.detail, ""]
    return "\n".join(out)
