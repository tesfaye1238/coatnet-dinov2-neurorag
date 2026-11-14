"""RAG evaluation metrics with explicit definitions.

Retrieval (binary relevance):
  Recall@k   = mean over queries of |relevant in top-k| / |relevant|   (queries with no relevant passage excluded and counted)
  HitRate@k  = fraction of queries with >= 1 relevant passage in top-k
  MRR@10     = mean reciprocal rank of the first relevant passage within the top 10 (0 if none)
  nDCG@10    = DCG/IDCG with gain 1 for relevant, discount 1/log2(rank+1)
Generation:
  BLEU-4     = corpus BLEU (clipped 1-4 gram precisions, brevity penalty, no smoothing); tokens: lowercase \\w+|punctuation
  ROUGE-L    = mean sentence-level LCS F1 on the same tokens
  BERTScore  = optional (needs the 'bert-score' package); versions and checkpoint are recorded by the caller
Claim-level factuality is computed from human annotations only; this module never labels a claim itself.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

CLAIM_LABELS = ("supported", "unverifiable", "contradicted", "unassessable")


def retrieval_metrics(runs: dict[str, list[str]], qrels: dict[str, set[str]], ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    qids = [q for q in runs if qrels.get(q)]
    skipped = [q for q in runs if not qrels.get(q)]
    out: dict = {"n_queries": len(qids), "n_queries_without_relevant": len(skipped)}
    if not qids:
        return {**out, **{f"recall@{k}": float("nan") for k in ks}}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean([len(set(runs[q][:k]) & qrels[q]) / len(qrels[q]) for q in qids]))
        out[f"hitrate@{k}"] = float(np.mean([bool(set(runs[q][:k]) & qrels[q]) for q in qids]))
    rr, nd = [], []
    for q in qids:
        top = runs[q][:10]
        rank = next((i + 1 for i, d in enumerate(top) if d in qrels[q]), None)
        rr.append(1.0 / rank if rank else 0.0)
        dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(top) if d in qrels[q])
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(qrels[q]), 10)))
        nd.append(dcg / idcg if idcg else 0.0)
    out["mrr@10"], out["ndcg@10"] = float(np.mean(rr)), float(np.mean(nd))
    return out


def tokenize(s: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", s.lower())


def _ngrams(t: list[str], n: int) -> Counter:
    return Counter(tuple(t[i : i + n]) for i in range(len(t) - n + 1))


def bleu4(candidates: list[str], references: list[str]) -> float:
    """Corpus BLEU-4, one reference per candidate, no smoothing (0.0 if any n-gram order has zero matches)."""
    if len(candidates) != len(references):
        raise ValueError("candidates and references must have equal length")
    match, total, clen, rlen = [0] * 4, [0] * 4, 0, 0
    for c, r in zip(candidates, references):
        ct, rt = tokenize(c), tokenize(r)
        clen += len(ct)
        rlen += len(rt)
        for n in range(1, 5):
            cg, rg = _ngrams(ct, n), _ngrams(rt, n)
            match[n - 1] += sum(min(v, rg[g]) for g, v in cg.items())
            total[n - 1] += max(len(ct) - n + 1, 0)
    if clen == 0 or min(match) == 0:
        return 0.0
    logp = sum(math.log(m / t) for m, t in zip(match, total)) / 4
    bp = 1.0 if clen > rlen else math.exp(1 - rlen / clen)
    return float(bp * math.exp(logp))


def _lcs(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def rouge_l(candidates: list[str], references: list[str]) -> float:
    scores = []
    for c, r in zip(candidates, references):
        ct, rt = tokenize(c), tokenize(r)
        if not ct or not rt:
            scores.append(0.0)
            continue
        lcs = _lcs(ct, rt)
        p, rec = lcs / len(ct), lcs / len(rt)
        scores.append(0.0 if lcs == 0 else 2 * p * rec / (p + rec))
    return float(np.mean(scores)) if scores else float("nan")


def bertscore_f1(candidates: list[str], references: list[str], model_type: str = "roberta-large", rescale: bool = False) -> dict:
    try:
        import bert_score
        from bert_score import score
    except ImportError as exc:
        raise ImportError("bert-score is required: pip install 'neurofusion[rag-eval]'") from exc
    _, _, f1 = score(candidates, references, model_type=model_type, rescale_with_baseline=rescale, lang="en")
    return {"f1_mean": float(f1.mean()), "model_type": model_type, "rescale_with_baseline": rescale, "bert_score_version": bert_score.__version__}


def claim_summary(labels: list[str]) -> dict:
    """Counts and fractions with explicit denominators. Unassessable claims are excluded from the assessed denominator."""
    bad = set(labels) - set(CLAIM_LABELS)
    if bad:
        raise ValueError(f"unknown claim labels {sorted(bad)}")
    c = Counter(labels)
    assessed = c["supported"] + c["unverifiable"] + c["contradicted"]
    f = lambda k, d: (c[k] / d) if d else float("nan")  # noqa: E731
    return {
        "counts": {k: c[k] for k in CLAIM_LABELS},
        "n_total": len(labels), "n_assessed": assessed,
        "supported_fraction": f("supported", assessed),
        "unverifiable_fraction": f("unverifiable", assessed),
        "contradicted_fraction": f("contradicted", assessed),
        "non_supported_fraction": ((c["unverifiable"] + c["contradicted"]) / assessed) if assessed else float("nan"),
        "denominator": "assessed claims (supported + unverifiable + contradicted); unassessable excluded",
    }
