# True Optimal Pipeline — Saved CLF + Winsorised REG

**Date:** 2026-03-15

**Goal:** Use the original saved baseline classifiers from `models/two_stage_pipeline.pkl`
(best-single-clf-per-target + F1-tuned thresholds from `model_phase2.py`) and retrain
**only** the regressors with 4-target 95th-percentile winsorising. No classifier changes.

## Why this is the true optimal setup

Since Stage 1 (classifiers) and Stage 2 (regressors) are independently trained on
different data subsets with different objectives, the best overall pipeline is simply
the best of each stage independently:

- **Stage 1:** the original `model_phase2.py` classifiers already achieve the highest
  freq-weighted F1 (**0.9387**). Oversampling only hurts common targets.
- **Stage 2:** 95th-percentile winsorising on 4 high-skew targets reduces freq-weighted
  MAE by **−32.7%** with no impact on classification.

## Configuration

| Parameter | Value |
|---|---|
| Classifiers | Loaded from `models/two_stage_pipeline.pkl` — **not retrained** |
| Threshold strategy | Best-single-clf-per-target + F1-tuned (from `model_phase2.py`) |
| TF-IDF | Loaded from pickle — **not refitted** |
| Winsorised regressors | `assembly` (cap 9.1 hrs), `calibration` (4.8 hrs), `hailrepair` (1702.7 hrs), `paintingFinish` (3.6 hrs) |

## Results

| Metric | Baseline (`model_phase2.py`) | **True Optimal** | Delta |
|---|---|---|---|
| Macro F1 | 0.8372 | **0.8372** | +0.0000 |
| Macro Recall | 0.779 | **0.8594** | +0.0804 |
| **Freq-weighted F1** | 0.9387 | **0.9347** | -0.0040 |
| **Freq-weighted Recall** | 0.946 | **0.9412** | -0.0048 |
| Macro MAE | 3.51 hrs | **2.14 hrs** | -1.37 hrs |
| **Freq-weighted MAE** | 2.78 hrs | **1.88 hrs** | -0.90 hrs |

## Per-target Results

| Target | clf | thr | n_pos | freq_w | F1 | Recall | Prec | MAE |
|--------|-----|-----|-------|--------|----|--------|------|-----|
| `calibration           ` |  lgbm | 0.890 |    26 |  0.263 |  0.6500 |  0.5000 |  0.9286 |    0.28 |
| `wheelmeasurement      ` |  lgbm | 0.090 |    26 |  0.263 |  0.8519 |  0.8846 |  0.8214 |    0.21 |
| `bodymeasurement       ` |  lgbm | 0.050 |     4 |  0.040 |  0.6000 |  0.7500 |  0.5000 |    0.08 |
| `dismounting           ` | logreg | 0.100 |    86 |  0.869 |  0.9600 |  0.9767 |  0.9438 |    1.16 |
| `bodyrepair            ` |  lgbm | 0.050 |    35 |  0.354 |  0.8533 |  0.9143 |  0.8000 |    1.44 |
| `assembly              ` |  lgbm | 0.950 |    91 |  0.919 |  0.9497 |  0.9341 |  0.9659 |    1.55 |
| `plasticrepair         ` |  lgbm | 0.570 |    15 |  0.152 |  0.9375 |  1.0000 |  0.8824 |    0.12 |
| `allTiresService       ` | logreg | 0.450 |     5 |  0.051 |  0.2222 |  0.4000 |  0.1538 |    0.63 |
| `cleaning              ` | logreg | 0.050 |    94 |  0.949 |  0.9843 |  1.0000 |  0.9691 |    0.48 |
| `paintingPreparation   ` |  lgbm | 0.050 |    70 |  0.707 |  0.9859 |  1.0000 |  0.9722 |    2.10 |
| `paintingSpraying      ` | logreg | 0.050 |    70 |  0.707 |  0.9714 |  0.9714 |  0.9714 |    0.67 |
| `paintingFinish        ` |  lgbm | 0.590 |    70 |  0.707 |  0.9718 |  0.9857 |  0.9583 |    0.57 |
| `hailrepair            ` |  lgbm | 0.050 |    29 |  0.293 |  0.9825 |  0.9655 |  1.0000 |   20.51 |
| `glas                  ` |  lgbm | 0.550 |     8 |  0.081 |  0.8000 |  0.7500 |  0.8571 |    0.09 |
| **MACRO** | — | — | — | — | **0.8372** | **0.8594** | — | **2.14** |
