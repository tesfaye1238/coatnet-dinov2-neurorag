"""Structured-output validation and passage-injection screening.

Three different things are kept apart:
 1. citation EXISTENCE  - a cited passage id was actually supplied (checked here);
 2. citation SUPPORT    - the passage supports the claim (NOT checked here; needs entailment or expert review);
 3. medical CORRECTNESS - independent clinical validity (NOT checked by any software here).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from neurofusion.rag.retrieval import Retrieved

REQUIRED_KEYS = ("predicted_category", "general_background", "contextual_summary", "evidence_limitations", "references_used")

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the )?(system|previous) (prompt|instructions)",
    r"you are now\b",
    r"(reveal|print|show) (the |your )?(system )?prompt",
    r"</?(system|assistant|passage)\b",
    r"\bact as\b.*\b(doctor|physician|oncologist)\b",
    r"new instructions?:",
    r"BEGIN (SYSTEM|INSTRUCTIONS)",
]
_INJ = [re.compile(p, re.I) for p in INJECTION_PATTERNS]

# Heuristic markers of individualised treatment advice. A warning, not proof.
ADVICE_PATTERNS = [re.compile(p, re.I) for p in (
    r"\byou should (start|stop|take|undergo|begin|discontinue)\b",
    r"\bI recommend\b", r"\bstart (chemotherapy|radiotherapy|treatment)\b",
    r"\bthe patient should\b", r"\bprescribe\b",
)]


def injection_flags(text: str) -> list[str]:
    return [p.pattern for p in _INJ if p.search(text)]


def screen_passages(passages: list[Retrieved], exclude_flagged: bool = True) -> tuple[list[Retrieved], dict[str, list[str]]]:
    flagged = {p.passage_id: injection_flags(p.text) for p in passages}
    flagged = {k: v for k, v in flagged.items() if v}
    kept = [p for p in passages if not (exclude_flagged and p.passage_id in flagged)]
    return kept, flagged


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    uncited_statements: int = 0
    citation_existence_checked: bool = True
    citation_support_assessed: bool = False
    medical_correctness_assessed: bool = False


class OutputParseError(ValueError):
    pass


def parse_json_output(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        raise OutputParseError("no JSON object found in model output")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise OutputParseError(f"invalid JSON: {exc}") from exc


def validate_output(out: dict, passages: list[Retrieved], predicted_category_display: str) -> ValidationReport:
    errors, warnings = [], []
    missing = [k for k in REQUIRED_KEYS if k not in out]
    if missing:
        return ValidationReport(False, [f"missing keys: {missing}"])
    supplied = {p.passage_id for p in passages}
    cited: set[str] = set()
    uncited = 0
    summary = out["contextual_summary"]
    if not isinstance(summary, list):
        return ValidationReport(False, ["contextual_summary must be a list"])
    for i, item in enumerate(summary):
        if not isinstance(item, dict) or "statement" not in item or "passage_ids" not in item:
            errors.append(f"contextual_summary[{i}] must have statement and passage_ids")
            continue
        ids = item["passage_ids"]
        if not ids:
            uncited += 1
        for pid in ids:
            if pid not in supplied:
                errors.append(f"contextual_summary[{i}] cites unknown passage id {pid!r}")
            cited.add(pid)
    refs = set(out["references_used"])
    unknown_refs = refs - supplied
    if unknown_refs:
        errors.append(f"references_used contains unknown ids {sorted(unknown_refs)}")
    if cited - refs:
        warnings.append(f"cited ids missing from references_used: {sorted(cited - refs)}")
    if uncited:
        warnings.append(f"{uncited} summary statement(s) carry no passage citation")
    if str(out["predicted_category"]).strip().lower() != predicted_category_display.lower():
        errors.append(f"predicted_category {out['predicted_category']!r} != classifier output {predicted_category_display!r}")
    blob = json.dumps(out)
    for rx in ADVICE_PATTERNS:
        if rx.search(blob):
            warnings.append(f"possible individualised-advice wording: /{rx.pattern}/")
    return ValidationReport(not errors, errors, warnings, sorted(cited), uncited)
