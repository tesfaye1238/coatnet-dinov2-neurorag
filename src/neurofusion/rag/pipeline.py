"""End-to-end NeuroRAG flow: query -> retrieve -> screen -> generate -> validate -> safeguard."""
from __future__ import annotations

import time

from neurofusion.rag.generation import Generator, no_retrieval_prompt
from neurofusion.rag.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from neurofusion.rag.retrieval import DISPLAY, Retriever
from neurofusion.rag.safeguards import (
    INSUFFICIENT,
    SAFEGUARD_PROVENANCE,
    UNCERTAIN,
    SafeguardPolicy,
    withheld_output,
)
from neurofusion.rag.validation import OutputParseError, parse_json_output, screen_passages, validate_output


class NeuroRAGPipeline:
    def __init__(self, retriever: Retriever | None, generator: Generator, policy: SafeguardPolicy | None = None):
        self.retriever, self.generator, self.policy = retriever, generator, policy or SafeguardPolicy()

    def run(self, category: str, confidence: float | None = None) -> dict:
        t0 = time.perf_counter()
        disp = DISPLAY.get(category, category)
        record: dict = {"category": category, "confidence": confidence, "prompt_version": PROMPT_VERSION,
                        "policy": self.policy.as_dict(), "safeguard_provenance": SAFEGUARD_PROVENANCE,
                        "generator": self.generator.describe()}
        passages, flagged, query = [], {}, None
        if self.retriever is not None:
            query, passages = self.retriever.retrieve(category, confidence)
            if self.policy.min_retrieval_score is not None:
                passages = [p for p in passages if p.score >= self.policy.min_retrieval_score]
            passages, flagged = screen_passages(passages, self.policy.exclude_injection_flagged)
            record["retrieval_config"] = self.retriever.cfg.as_dict()
            record["embedder"] = self.retriever.embedder.describe()
        record.update(query=query, injection_flagged=flagged, retrieved=[{"passage_id": p.passage_id, "score": p.score, "rank": p.rank} for p in passages])
        if self.retriever is not None and len(passages) < self.policy.min_passages:
            record.update(output=withheld_output(disp, INSUFFICIENT), validation=None, status="insufficient_evidence")
            record["latency_s"] = time.perf_counter() - t0
            return record
        user = build_user_prompt(category, confidence, passages) if self.retriever is not None else no_retrieval_prompt(category, confidence)
        raw = self.generator.generate(SYSTEM_PROMPT, user)
        record["raw_output"] = raw
        try:
            out = parse_json_output(raw)
            rep = validate_output(out, passages, disp)
        except OutputParseError as exc:
            out, rep = None, None
            record["parse_error"] = str(exc)
        if rep is None or (not rep.ok and self.policy.withhold_on_invalid):
            record.update(output=withheld_output(disp, "model output failed validation"), validation=rep.__dict__ if rep else None, status="withheld")
        else:
            thr = self.policy.confidence_threshold
            if thr is not None and confidence is not None and confidence < thr:
                out["evidence_limitations"] = (out.get("evidence_limitations", "") + " " + UNCERTAIN).strip()
            record.update(output=out, validation=rep.__dict__, status="ok" if rep.ok else "invalid_not_withheld")
        record["latency_s"] = time.perf_counter() - t0
        return record
