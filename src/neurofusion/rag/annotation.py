"""Expert-review support: export sheets, import annotations, agreement. The software never invents ratings."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from neurofusion.rag.evaluation import CLAIM_LABELS

SHEET_COLUMNS = ["query_id", "blinded_system_id", "claim_id", "claim_text", "cited_passage_ids", "reviewer_code", "label", "comment"]
REVIEWER_CODE = re.compile(r"^R\d{2}$")


class AnnotationError(ValueError):
    pass


def export_claim_sheet(rows: list[dict], path: str | Path) -> Path:
    """rows: dicts with query_id, blinded_system_id, claim_id, claim_text, cited_passage_ids. Labels left blank."""
    df = pd.DataFrame(rows)
    for c in SHEET_COLUMNS:
        if c not in df:
            df[c] = ""
    df = df[SHEET_COLUMNS]
    df["reviewer_code"], df["label"] = "", ""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def import_annotations(path: str | Path, allow_blank: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in SHEET_COLUMNS if c not in df.columns]
    if missing:
        raise AnnotationError(f"annotation file missing columns {missing}")
    blank = df["label"] == ""
    if blank.any() and not allow_blank:
        raise AnnotationError(f"{int(blank.sum())} claim(s) have no label")
    bad = set(df.loc[~blank, "label"]) - set(CLAIM_LABELS)
    if bad:
        raise AnnotationError(f"unknown labels {sorted(bad)}; allowed {CLAIM_LABELS}")
    bad_codes = [c for c in df["reviewer_code"].unique() if c and not REVIEWER_CODE.match(c)]
    if bad_codes:
        raise AnnotationError(f"reviewer codes must look like R01 (non-identifying); got {bad_codes}")
    return df


def agreement(df: pd.DataFrame, reviewer_a: str, reviewer_b: str) -> dict:
    a = df[df["reviewer_code"] == reviewer_a].set_index("claim_id")["label"]
    b = df[df["reviewer_code"] == reviewer_b].set_index("claim_id")["label"]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        raise AnnotationError("no claims were labelled by both reviewers")
    la, lb = a.loc[common], b.loc[common]
    return {
        "n_common": int(len(common)),
        "percent_agreement": float((la == lb).mean()),
        "cohen_kappa": float(cohen_kappa_score(la, lb)) if la.nunique() > 1 or lb.nunique() > 1 else float("nan"),
        "disagreements_for_adjudication": sorted(common[(la != lb).values].tolist()),
    }
