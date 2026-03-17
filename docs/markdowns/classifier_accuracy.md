# Classifier Accuracy Metrics

**Date:** 2026-03-15
**Model:** Baseline Classifiers (`models/two_stage_pipeline.pkl`)

This report details the **Classification Accuracy** for the 14 targets. Accuracy is evaluated on the standard test split (20% holdout, seeded, stratified) using the best single classifier per target and its F1-tuned threshold.

## Why look at Accuracy?

Accuracy measures the subset of orders where the model's binary prediction (0 or 1) perfectly matches the ground truth. While intuitive, it must be interpreted carefully for our highly imbalanced data.

For very rare targets (like `allTiresService` or `bodymeasurement`), the model could simply predict `0` for every order and still achieve >90% accuracy. This is why **F1 Score** and **Recall** are our primary optimization goals — they penalise the model heavily when it fails to identify positive instances, forcing it to actually "learn" the rare targets rather than guessing the negative class.

The **Frequency-weighted Accuracy** acts as a sanity check, showing how often the pipeline is "right overall" when adjusting for how common a work step actually is in production.

## Aggregates

| Metric | Score |
|---|---|
| **Macro Accuracy** (unweighted) | **0.9380** |
| **Freq-weighted Accuracy** | **0.9433** |
| Macro F1 (unweighted) | 0.8372 |
| Freq-weighted F1 | 0.9387 |

## Per-target Results

| Target | freq_weight | n_test_pos | **Accuracy** | F1 | Recall |
|--------|-------------|------------|--------------|----|--------|
| `cleaning` | 0.949 | 94 | **0.9697** | 0.9843 | 1.0000 |
| `assembly` | 0.919 | 91 | **0.9091** | 0.9497 | 0.9341 |
| `dismounting` | 0.869 | 86 | **0.9293** | 0.9600 | 0.9767 |
| `paintingSpraying` | 0.707 | 70 | **0.9596** | 0.9714 | 0.9714 |
| `paintingFinish` | 0.707 | 70 | **0.9596** | 0.9718 | 0.9857 |
| `paintingPreparation` | 0.707 | 70 | **0.9798** | 0.9859 | 1.0000 |
| `bodyrepair` | 0.354 | 35 | **0.8889** | 0.8533 | 0.9143 |
| `hailrepair` | 0.293 | 29 | **0.9899** | 0.9825 | 0.9655 |
| `calibration` | 0.263 | 26 | **0.8586** | 0.6500 | 0.5000 |
| `wheelmeasurement` | 0.263 | 26 | **0.9192** | 0.8519 | 0.8846 |
| `plasticrepair` | 0.152 | 15 | **0.9798** | 0.9375 | 1.0000 |
| `glas` | 0.081 | 8 | **0.9697** | 0.8000 | 0.7500 |
| `allTiresService` | 0.051 | 5 | **0.8586** | 0.2222 | 0.4000 |
| `bodymeasurement` | 0.040 | 4 | **0.9596** | 0.6000 | 0.7500 |

*Targets are sorted by `freq_weight` descending to highlight the most common work steps.*
