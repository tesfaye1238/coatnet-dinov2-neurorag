# Reviewer concerns -> code -> evidence still required

No reviewer correspondence was supplied; the concerns below are the issues listed in the task brief. **Software existing does not resolve a
concern.** "Open" means the stated evidence is still missing.

| Concern | Code | Evidence required to close it | Status |
|---|---|---|---|
| Conflicting precision/recall/F1 in the text | `numeric_audit` A05-A08, `metrics.classification_report` | Corrected text from the saved test predictions | Open (audit shows Table 4 and Figure 8 agree; narrative does not) |
| Baseline accuracies incompatible with 1,600 images | A03, A15 | Sample-level predictions for each baseline | Open |
| Balanced single-label test set: macro-recall = accuracy | A08 (checked), `metrics` | Predictions | Open for the narrative |
| Test vs validation curves | `Trainer` logs only train/val | Re-plotted validation curves from a real run | Open |
| Patient independence not established | `grouping`, `splits.audit_split` | Verified patient ids for all images, patient-disjoint split, or a re-evaluation | Open (figshare only) |
| Cross-validation counts | `splits.kfold_splits` | Original fold manifests | Open |
| Source-held-out validity | `check_source_compatibility`, label rejection | Per-image source ids, compatible 4-class labels, cross-source duplicate audit | Open (D04) |
| RAG settings proposed, not confirmed | `configs/rag.yaml`, `SAFEGUARD_PROVENANCE` | Original configuration and code | Open |
| Reported CIs and paired tests need procedures | `bootstrap`, `paired` | Original predictions and procedure, or a full recomputation | Open (D08-D10) |
| Unsupported generation / hallucination measurement | `validation`, `evaluation.claim_summary`, `annotation` | Annotation protocol, claim-level labels from qualified reviewers, agreement | Open; no ratings exist |
| Reproducibility artefacts (code, configs, seeds, versions) | repository, `run_record.json`, configs | A trained run with its record and a verified archive | Partly: tooling exists, no run |
| Overclaim of interpretability from selected examples | `gradcam` docs, app labels | Maps over all test images incl. errors and no-tumor, expert masks if any | Open |
| Usability / clinical value of generated summaries | none | Prospective study | Not addressed |
