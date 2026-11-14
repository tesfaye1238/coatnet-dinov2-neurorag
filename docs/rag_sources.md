# NeuroRAG reference collection and retrieval

The indexed count always comes from the real collection; nothing is invented to match the manuscript's 1,842 entries. Source documents and
chunks are counted separately (`rag-build` prints `source_documents`, `chunks`, effective chunk token lengths).

## Source manifest (`source_manifest.csv` or `.json`)

Required fields: `doc_id, title, issuer, url_or_doi, version_date, access_date, license, sha256, category_tags, review_status, path`.
`category_tags` are `;`-separated from `glioma, meningioma, notumor, pituitary, general`. `review_status` is `approved | pending | rejected |
demo_synthetic`; only `approved` is indexed (demo documents only with `--allow-demo`). `rag-build` verifies every file's sha256 and stops if any
file is missing or altered. Importers: `.txt`, `.md`, `.html`, `.pdf` (pypdf), `.json`.

The shipped `rag-demo-collection` writes **8 synthetic passages for software tests**. They are not sourced references, not a reconstruction of
the manuscript's collection, and must not be cited or used clinically.

## Candidate settings (not verified as the historical ones; `configs/rag.yaml`)

BAAI/bge-base-en-v1.5 (768-d, normalised, query instruction recorded, revision to be pinned); cosine similarity; 512-token chunks including
special tokens with 64 overlap, checked with the model's own fast tokenizer (offset mapping) so passages are never silently truncated and
`ChunkLengthError` is raised instead; top-10 candidates, top-5 to generation; optional category metadata filter; no reranker.

`HashingEmbedder` is a deterministic lexical stand-in for tests and demos and is labelled non-semantic. Do not call a BGE configuration
"NeuroRAG" without stating how it differs from the BGE baseline in Table 8A; that difference is not described in the manuscript.

## Stores

`NumpyStore` (dependency-free, saved as `vectors.npy` + `meta.json`) and `QdrantStore` (memory, local path or URL). A test asserts both give the
same ranking and that category filtering works.

## Generation and safeguards

Qwen is configured, not assumed: `Qwen/Qwen2.5-7B-Instruct` is a **candidate** checkpoint, with optional revision pin, dtype, device, optional
documented quantisation, greedy decoding, max 512 new tokens and a context-token budget (over-budget prompts raise; no silent truncation).
The system prompt is the informational-scope prompt of the task brief and treats passages as data. Safeguards and their origin:

| Safeguard | Origin |
|---|---|
| retrieval grounding; task-constrained prompting | reported in the manuscript |
| passage-id existence check; JSON schema validation; injection screening of passages; insufficient-evidence response; uncertainty statement below a configurable threshold; withhold on validation failure; advice-wording heuristic | new in this repository |

Checking that a cited passage **exists** is not the same as checking that it **supports** the claim, nor that the claim is medically correct;
the validator records `citation_support_assessed = false` and `medical_correctness_assessed = false`. Any confidence threshold must be chosen on
development data and is not claimed to be clinically validated.
