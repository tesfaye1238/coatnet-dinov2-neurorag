# RAG evaluation and clinical review support

A 200-query evaluation is a **plan** until the queries, reference summaries, system outputs and assessments exist. Nothing here produces them.

## Schema (one JSON/CSV record per query-system pair)

`query_id, category, relevant_passage_ids, retrieved_passage_ids + scores, system_output, reference_summary (independently prepared),
claims[claim_id, text, cited_passage_ids, label, reviewer_code], timing, config (embedder, revision, chunking, k, prompt version, generator,
decoding)`. Reviewers are non-identifying codes `R01`, `R02`, ...; the importer rejects anything else.

## Retrieval metrics (`rag/evaluation.py`)

Binary relevance. Recall@k = mean of |relevant in top-k| / |relevant|; HitRate@k = share of queries with at least one relevant passage;
MRR@10; nDCG@10 (gain 1, discount 1/log2(rank+1)). Queries without relevant passages are excluded and counted.

## Generation metrics

BLEU-4 (corpus, clipped n-grams, brevity penalty, no smoothing, lower-cased `\w+|punctuation` tokens), ROUGE-L (mean sentence LCS F1) and
optional BERTScore (needs `bert-score`; model type and version are returned). Report the tokenisation and checkpoints with any number.

## Comparisons

Compare (1) Qwen2.5 without retrieval (`no_retrieval_prompt`), (2) a clearly defined naive top-k retrieval (`category_filter: false`, no
screening), (3) the implemented configuration, on the same queries and generation settings.

## Expert review

`export_claim_sheet` writes a sheet with blank labels; `import_annotations` refuses blanks, unknown labels and identifying reviewer codes;
`agreement` gives percent agreement, Cohen's kappa and the claims needing adjudication. The software never invents a rating.

## Claim categories and denominators

Labels: supported, unverifiable, contradicted (fabricated), unassessable. Fractions are over **assessed** claims (the first three); unassessable
claims are excluded and counted. If the counts 847 / 68 / 85 are supplied, the contradicted fraction is 85/1000 = 8.5% and the combined
non-supported fraction is 153/1000 = 15.3%; these are different quantities, and the arithmetic does not verify how the annotations were made.
A custom claim-level measure must not be called FActScore unless that method is actually implemented.
