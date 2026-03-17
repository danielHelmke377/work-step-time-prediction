# Fully Mixed Pipeline — Best-Per-Target CLF + Best-Per-Target REG

**Date:** 2026-03-17

**Goal:** Use the best classifier and the best regressor independently per target,
both chosen by validation-set comparison. No oversampling, class weights only.

## Classifier Summary

| Model | n targets |
|---|---|
| Logistic Regression | 4 |
| LightGBM | 10 |

## Comparison vs Previous Experiments

| Metric | Baseline | True Optimal | Best-per-REG | **Fully Mixed** |
|---|---|---|---|---|
| Macro F1 | 0.8372 | 0.8372 | 0.8372 | **0.8396** |
| Macro Recall | 0.779 | 0.8594 | 0.8594 | **0.8594** |
| Macro Accuracy | 0.9380 | 0.9380 | 0.9380 | **0.9401** |
| Freq-weighted F1 | 0.9387 | 0.9347 | 0.9347 | **0.9357** |
| Freq-weighted Recall | 0.9460 | 0.9412 | 0.9412 | **0.9412** |
| Freq-weighted Accuracy | 0.9433 | 0.9433 | 0.9433 | **0.9449** |
| Macro MAE | 3.51 hrs | 2.14 hrs | 0.70 hrs | **0.7 hrs** |
| **Freq-weighted MAE** | 2.78 hrs | 1.88 hrs | 0.96 hrs | **0.96 hrs** |

## Per-target Results

| Target | clf | reg | n_pos | freq_w | Accuracy | F1 | Recall | MAE |
|--------|-----|-----|-------|--------|----------|----|--------|-----|
| `calibration` | lgbm | lgbm | 26 | 0.263 | 0.8586 | 0.6500 | 0.5000 | 0.39 |
| `wheelmeasurement` | lgbm | lgbm | 26 | 0.263 | 0.9192 | 0.8519 | 0.8846 | 0.21 |
| `bodymeasurement` | lgbm | lgbm | 4 | 0.040 | 0.9596 | 0.6000 | 0.7500 | 0.08 |
| `dismounting` | logreg | lgbm | 86 | 0.869 | 0.9394 | 0.9655 | 0.9767 | 1.15 |
| `bodyrepair` | lgbm | ridge | 35 | 0.354 | 0.8889 | 0.8533 | 0.9143 | 1.73 |
| `assembly` | lgbm | ridge | 91 | 0.919 | 0.9091 | 0.9497 | 0.9341 | 1.59 |
| `plasticrepair` | lgbm | lgbm | 15 | 0.152 | 0.9798 | 0.9375 | 1.0000 | 0.12 |
| `allTiresService` | logreg | lgbm | 5 | 0.051 | 0.8788 | 0.2500 | 0.4000 | 0.55 |
| `cleaning` | logreg | lgbm | 94 | 0.949 | 0.9697 | 0.9843 | 1.0000 | 0.48 |
| `paintingPreparation` | lgbm | ridge | 70 | 0.707 | 0.9798 | 0.9859 | 1.0000 | 1.91 |
| `paintingSpraying` | logreg | lgbm | 70 | 0.707 | 0.9596 | 0.9714 | 0.9714 | 0.67 |
| `paintingFinish` | lgbm | ridge | 70 | 0.707 | 0.9596 | 0.9718 | 0.9857 | 0.77 |
| `hailrepair` | lgbm | ridge | 29 | 0.293 | 0.9899 | 0.9825 | 0.9655 | 0.02 |
| `glas` | lgbm | lgbm | 8 | 0.081 | 0.9697 | 0.8000 | 0.7500 | 0.09 |
| **MACRO** | | | | | **0.9401** | **0.8396** | **0.8594** | **0.7** |
| **FREQ-WT** | | | | | **0.9449** | **0.9357** | **0.9412** | **0.96** |
