"""Safeguard policy and provenance of each safeguard (reported in the manuscript vs newly implemented here)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

SAFEGUARD_PROVENANCE = {
    "retrieval_grounding": "reported_in_manuscript",
    "task_constrained_prompting": "reported_in_manuscript",
    "passage_id_existence_check": "new_in_this_repository",
    "structured_output_schema_validation": "new_in_this_repository",
    "prompt_injection_screening_of_passages": "new_in_this_repository",
    "insufficient_evidence_response": "new_in_this_repository",
    "uncertainty_statement_below_threshold": "new_in_this_repository",
    "withhold_on_validation_failure": "new_in_this_repository",
    "individualised_advice_wording_heuristic": "new_in_this_repository",
}


@dataclass
class SafeguardPolicy:
    min_passages: int = 1
    min_retrieval_score: float | None = None
    confidence_threshold: float | None = None  # None = no uncertainty statement; any value must come from DEVELOPMENT data
    exclude_injection_flagged: bool = True
    withhold_on_invalid: bool = True

    def as_dict(self) -> dict:
        return {**asdict(self), "threshold_note": "not clinically validated; select on development data only"}


INSUFFICIENT = "Evidence is insufficient: too few reliable reference passages were retrieved to support a contextual summary."
UNCERTAIN = "The classifier's confidence for this category is below the configured threshold; the category may be wrong, and the background below should be read with that in mind."


def withheld_output(category_display: str, reason: str) -> dict:
    return {"predicted_category": category_display, "general_background": "", "contextual_summary": [],
            "evidence_limitations": f"Output withheld: {reason}", "references_used": [], "withheld": True}
