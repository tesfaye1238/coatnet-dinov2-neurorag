"""A tiny SYNTHETIC demonstration collection for software tests and the local demo.

These passages were written for software testing. They are NOT sourced medical references, are not a
reconstruction of the manuscript's reference collection, and must never be cited or relied on clinically.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from neurofusion.utils.hashing import sha256_bytes

DEMO_TEXTS = {
    "demo-glioma-01": ("glioma", "Gliomas are tumours that arise from glial cells of the brain. Imaging characteristics vary, and a category assigned by an image classifier is not a pathological diagnosis."),
    "demo-glioma-02": ("glioma", "Management decisions for gliomas depend on tissue diagnosis, grade and molecular features, which an image category alone cannot provide."),
    "demo-meningioma-01": ("meningioma", "Meningiomas arise from the meninges, the membranes surrounding the brain and spinal cord. Many are slow growing, and clinical context determines follow-up."),
    "demo-meningioma-02": ("meningioma", "A meningioma category from an image classifier does not establish tumour grade or the need for any specific intervention."),
    "demo-pituitary-01": ("pituitary", "Pituitary tumours arise in the pituitary gland at the base of the brain. Hormonal evaluation is part of clinical assessment and is not captured by an image category."),
    "demo-pituitary-02": ("pituitary", "A pituitary category from an image classifier is informational and does not replace endocrine or ophthalmic evaluation."),
    "demo-notumor-01": ("notumor", "A no-tumour prediction means the classifier assigned none of the three tumour categories. It does not establish the absence of other abnormalities."),
    "demo-notumor-02": ("notumor", "Symptoms or abnormal findings still require clinical assessment even when an image classifier outputs no tumour."),
}


def write_demo_collection(directory: str | Path) -> Path:
    d = Path(directory)
    (d / "docs").mkdir(parents=True, exist_ok=True)
    rows = []
    for doc_id, (cat, text) in DEMO_TEXTS.items():
        p = d / "docs" / f"{doc_id}.txt"
        p.write_text(text, encoding="utf-8")
        rows.append({
            "doc_id": doc_id, "title": f"SYNTHETIC DEMO PASSAGE {doc_id}", "issuer": "neurofusion software tests",
            "url_or_doi": "n/a (synthetic)", "version_date": "2026-11-14", "access_date": "2026-11-14",
            "license": "written for software testing; no clinical use", "sha256": sha256_bytes(text.encode("utf-8")),
            "category_tags": cat, "review_status": "demo_synthetic", "path": f"docs/{doc_id}.txt",
        })
    out = d / "source_manifest.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
