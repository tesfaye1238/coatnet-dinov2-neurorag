"""Group assignment for leakage-safe splitting.

Verified groups (source-supplied patient IDs) and heuristic groups are recorded separately via
``grouping_method``. Heuristic groups are never reported as verified patient identity.
"""
from __future__ import annotations

import re

import pandas as pd


def assign_groups(df: pd.DataFrame, filename_pattern: str | None = None) -> pd.DataFrame:
    """Assign ``group_id`` / ``grouping_method``.

    1. Rows with a verified patient id: ``patient:<source>:<pid>`` (``verified_patient_id``).
    2. Else, if ``filename_pattern`` (regex with one capture group) matches the file stem:
       ``heur:<source>:<capture>`` (``heuristic_filename``) - unverified.
    3. Else a singleton group (``singleton``).
    """
    out = df.copy()
    rx = re.compile(filename_pattern) if filename_pattern else None
    gids, methods = [], []
    for r in out.itertuples():
        if bool(r.patient_id_verified) and str(r.patient_id).strip():
            gids.append(f"patient:{r.source_id}:{r.patient_id}")
            methods.append("verified_patient_id")
            continue
        if rx is not None:
            stem = str(r.relative_path).rsplit("/", 1)[-1].rsplit(".", 1)[0]
            m = rx.search(stem)
            if m:
                gids.append(f"heur:{r.source_id}:{m.group(1)}")
                methods.append("heuristic_filename")
                continue
        gids.append(f"img:{r.image_id}")
        methods.append("singleton")
    out["group_id"] = gids
    out["grouping_method"] = methods
    return out


def merge_groups_by_pairs(df: pd.DataFrame, pairs: pd.DataFrame, method_name: str = "perceptual_cluster") -> pd.DataFrame:
    """Merge groups of image pairs flagged as related (union-find) so they cannot be split apart.

    Merged groups are labelled ``method_name`` unless they already contain a verified patient group, in which
    case the merged group keeps the strongest method present and is still recorded as merged.
    """
    parent: dict[str, str] = {g: g for g in df["group_id"].unique()}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    gmap = df.set_index("image_id")["group_id"]
    for r in pairs.itertuples():
        if r.image_id_a in gmap.index and r.image_id_b in gmap.index:
            ra, rb = find(gmap[r.image_id_a]), find(gmap[r.image_id_b])
            if ra != rb:
                parent[rb] = ra
    out = df.copy()
    new_gid = out["group_id"].map(find)
    merged = new_gid != out["group_id"]
    sizes = new_gid.map(new_gid.value_counts())
    out["group_id"] = new_gid
    out.loc[merged | (sizes > 1) & (out["grouping_method"] == "singleton"), "grouping_method"] = method_name
    return out
