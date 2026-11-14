# Evaluation protocol

1. Models are trained on the train split only; the best epoch is the lowest **validation** loss. The trainer has no test loader; test
   predictions are written once, after the best checkpoint is loaded. Training logs hold train and validation curves only.
2. Every metric is computed from saved sample-level predictions (`image_id, y_true, y_pred, p_<class>`), never from typed-in numbers.
   `load_predictions` rejects rows that do not sum to 1 or whose `y_pred` is not the argmax.
3. Metrics (definitions in the module docstrings): accuracy with correct/total; per-class and macro precision, recall and F1 (undefined
   values are NaN, listed in `undefined`, and macro averages state how many classes were used); confusion matrix; OvR specificity and OvR MCC
   (distinct from the multiclass MCC); OvR ROC-AUC and step-wise average precision (not trapezoidal PR area); ECE (15 equal-width bins), MCE,
   Brier, NLL, risk-coverage, AURC and coverage at 1% risk.
4. Uncertainty: percentile bootstrap, resampling clusters when verified patient clusters are supplied, otherwise images (labelled conditional on
   independent images). It reflects test-set sampling for a fixed model; training-seed variability is estimated separately with `seeds`.
   Seeds and repetitions are returned with every interval.
5. Paired comparisons join on `image_id`, require identical ground truth and report discordant counts. McNemar is reported uncorrected,
   continuity-corrected and exact; the exact p is recommended because discordant counts are small. Multiplicity is controlled only within a
   family named by the caller (Holm and Bonferroni). Macro-AUC differences use a paired bootstrap; no binary DeLong is applied to a multiclass
   quantity.
6. Source-held-out: audit cross-source duplicates and class availability; reject unmapped labels; absent classes are reported, not imputed.
7. Robustness severities follow Supplementary S2 where documented; the perturbations are image-space approximations, not MRI acquisition
   physics. Efficiency is measured only on the hardware it is run on; energy is not reported without a documented method.
8. `reconcile_accuracy` prints recomputed and reported values side by side and labels the run a new experiment. Differences are never used to
   adjust anything.
