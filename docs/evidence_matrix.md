# Evidence matrix

Maps manuscript content to implementation, the inputs it needs, how it is verified, and what is unresolved. "Status" describes this repository,
not the manuscript's claims.

| Manuscript | Implementation | Required inputs | Verification procedure | Status | Unresolved |
|---|---|---|---|---|---|
| 3.1, Table 1 dataset and splits | `data/sources.py`, `manifest.py`, `dedup.py`, `grouping.py`, `splits.py` | The three source collections; per-image manifest | `audit-manifest --expected` reports realised counts and differences from Table 1; `dedup` exports review pairs | Code and tests complete; **no data run** (figshare not downloaded; SARTAJ/Br35H need login) | Exact 7,022 images, split files, patient ids for two sources (D01-D04, D13) |
| 3.2 preprocessing, augmentation | `data/transforms.py` | none | Unit tests: deterministic eval transform, stochastic train transform, reported parameters | Complete; unreported details are config options (N) | Original order, fill, erasing ranges |
| 3.3, Table 2 architecture | `models/` | Pretrained weights | Dimension checks with real weights (768, 384, 1,152, 144); parameter audit | Verified dims; totals differ by 0.085 M | Authors' CoAtNet implementation |
| 5, Table 3 training | `training/` | Data, GPU for a full run | Schedule unit tests, resume, early stopping, validation-only best checkpoint | Code complete; **full training not executed** (CPU-only torch build, no data) | Unreported optimiser details |
| 5.1, 6.1-6.4 metrics | `evaluation/metrics.py` | Sample-level predictions | Tested against scikit-learn; audit recomputes from Figure 8 | Complete | Original predictions |
| Table 4 baselines | `models/baselines.py` | Weights, data | Instantiation tests; accuracy-attainability audit | Five baselines instantiate; ESPNetV2 unavailable | ESPNetV2 source; baseline predictions |
| 6.6, Table 5 ablations | `models/factory.py` | Data | Parameter audit; reported variants all constructible | Complete (no training run) | Training results |
| Table 5 / S4 intervals | `evaluation/bootstrap.py` | Predictions, patient ids | Determinism and cluster tests; compare with analytic intervals (A17) | Complete | Original procedure |
| S5 paired tests | `evaluation/paired.py` | Predictions of both models | Exact McNemar tests; audit infers discordant counts (A15) | Complete | Original predictions; multiclass DeLong (A16) |
| 6.5.1, Fig. 12 attribution | `explainability/gradcam.py` | Trained checkpoint | Gradient-flow, shape and zero-map tests; raw arrays saved | Complete; **no trained checkpoint, no figure produced** | Original maps |
| S1 calibration | `evaluation/calibration.py` | Predictions | Known-case tests | Complete | Original values |
| S2/S3 robustness | `evaluation/robustness.py` | Checkpoint, test images | Range, seeding and severity tests | Complete; not run on a model | Original functional forms |
| 6.6, Table 6 efficiency | `evaluation/efficiency.py` | Hardware | FLOP formula test; latency fields | Complete; measures on the local CPU only if run | Hardware, energy method |
| 7B source-held-out | `data/splits.py` | Per-image source ids and compatible labels | Class-availability audit, binary-label rejection | Complete | Compatible labels (D04) |
| 4.2, 8A retrieval | `rag/` (chunking, embedding, index, retrieval) | Source documents | Tokenizer-aware chunking tests; Qdrant == NumPy ranking | Complete with a synthetic demo | The 1,842-entry collection and its sources |
| 4.3 generation, S6 safeguards | `rag/generation.py`, `validation.py`, `safeguards.py`, `pipeline.py` | LLM weights for Qwen | Citation, injection, empty-context, withheld-output tests | Complete; Qwen path **not run** (7B weights not downloaded) | Original prompts and settings |
| 8B generation metrics | `rag/evaluation.py`, `annotation.py` | Queries, qrels, references, annotations | Hand-checked metric tests; claim denominators | Complete; **no evaluation data** | 200-query set (a plan only) |
| 6.6 prototype interface | `app_logic.py`, `app/streamlit_app.py` | Checkpoint | Upload-validation, missing-weights, AppTest | Complete | Usability not evaluated |
