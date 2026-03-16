# Markdowns — Documentation Index

This folder contains all written documentation and analysis reports for the
Repair Order Work Step Prediction project.
German-language originals are archived in [`de/`](de/).

---

## Model Documentation

| File | Description |
|---|---|
| [model_summary.md](model_summary.md) | **Start here.** Complete pipeline overview: problem framing, two-stage architecture (classifier → regressor), feature engineering, training setup, and performance summary. Written at senior data-scientist level. |
| [assessment_writeup.md](assessment_writeup.md) | Technical assessment solution brief covering problem framing, methodology, results, and production-readiness considerations. |
| [reflection.md](reflection.md) | Honest project reflection: what went well, what is uncertain, and known limitations. |
| [production_roadmap.md](production_roadmap.md) | Roadmap for moving the pipeline from prototype to production: serving, monitoring, retraining triggers, and required infrastructure. |

---

## Stage 1 — Classifiers

| File | Description |
|---|---|
| [classifier_selection.md](classifier_selection.md) | Comparison of classifier candidates (Logistic Regression vs. LightGBM), evaluation criteria, and rationale for the final selection per target. |
| [logreg_classifier_metrics.md](logreg_classifier_metrics.md) | Per-target performance report for the Logistic Regression classifier: F1, Precision, Recall, and threshold values across all 14 work-step targets. |
| [classifier_accuracy.md](classifier_accuracy.md) | Macro and frequency-weighted Accuracy for the baseline classifiers. Includes a per-target breakdown alongside F1 and Recall, and a note on why accuracy alone is insufficient for imbalanced targets. |
| [ngrams_explained.md](ngrams_explained.md) | Explains word-level and character-level N-gram TF-IDF features and why both are used together. |

---

## Stage 1 — Optimisation Experiments

| File | Description |
|---|---|
| [oversampling_results.md](oversampling_results.md) | Results from oversampling minority-class targets (SMOTE / random) to improve recall on rare work steps. |
| [oversample_recall_results.md](oversample_recall_results.md) | Combined results of oversampling with a Recall-floor ≥ 90% threshold strategy. Shows trade-off between Precision and Recall across strategies. |
| [optimisation_results.md](optimisation_results.md) | Results from soft-vote ensemble (LogReg + LightGBM) and F1-optimal threshold tuning experiments. |
| [optimisation_summary_report.md](optimisation_summary_report.md) | Summary report consolidating all Stage 1 optimisation experiments, with a final recommendation on which strategy to deploy. |

---

## Stage 2 — Regressors

| File | Description |
|---|---|
| [regressor_selection.md](regressor_selection.md) | Comparison of regression candidates (Linear, Ridge, LightGBM), evaluation via MAE per target, and rationale for the LGBMRegressor selection. |
| [rule_baseline_explainer.md](rule_baseline_explainer.md) | Explains the rule-based baseline (median duration per target), how it is computed, and how it is used as a benchmark for the ML regressor. |

---

## Subfolder

| Folder | Description |
|---|---|
| [`de/`](de/) | German-language originals of classifier and regressor selection documents, archived for reference. |
