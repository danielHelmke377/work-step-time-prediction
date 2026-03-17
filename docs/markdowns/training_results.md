# Fully Mixed Pipeline — Best-Per-Target CLF + Best-Per-Target REG

**Date:** 2026-03-17

**Goal:** Use the best classifier and the best regressor independently per target,
both chosen by validation-set comparison. No oversampling, class weights only.

## Classifier Summary

| Model | n targets |
|---|---|
| Logistic Regression | 0 |
| LightGBM | 14 |

## Comparison vs Previous Experiments

| Metric | Baseline | True Optimal | Best-per-REG | **Fully Mixed** |
|---|---|---|---|---|
| Macro F1 | 0.8372 | 0.8372 | 0.8372 | **0.2959** |
| Macro Recall | 0.779 | 0.8594 | 0.8594 | **0.4048** |
| Macro Accuracy | 0.9380 | 0.9380 | 0.9380 | **0.6543** |
| Freq-weighted F1 | 0.9387 | 0.9347 | 0.9347 | **0.358** |
| Freq-weighted Recall | 0.9460 | 0.9412 | 0.9412 | **0.4444** |
| Freq-weighted Accuracy | 0.9433 | 0.9433 | 0.9433 | **0.6657** |
| Macro MAE | 3.51 hrs | 2.14 hrs | 0.70 hrs | **1.72 hrs** |
| **Freq-weighted MAE** | 2.78 hrs | 1.88 hrs | 0.96 hrs | **1.89 hrs** |

## Per-target Results

| Target | clf | reg | n_pos | freq_w | Accuracy | F1 | Recall | MAE |
|--------|-----|-----|-------|--------|----------|----|--------|-----|
| `calibration` | lgbm | lgbm | 14 | 0.280 | 0.5200 | 0.4000 | 0.5714 | 2.13 |
| `wheelmeasurement` | lgbm | lgbm | 5 | 0.100 | 0.7600 | 0.1429 | 0.2000 | 0.17 |
| `bodymeasurement` | lgbm | lgbm | 8 | 0.160 | 0.5800 | 0.1600 | 0.2500 | 2.05 |
| `dismounting` | lgbm | lgbm | 7 | 0.140 | 0.6600 | 0.2609 | 0.4286 | 1.39 |
| `bodyrepair` | lgbm | ridge | 9 | 0.180 | 0.4800 | 0.3158 | 0.6667 | 2.63 |
| `assembly` | lgbm | ridge | 8 | 0.160 | 0.7400 | 0.4800 | 0.7500 | 2.08 |
| `plasticrepair` | lgbm | lgbm | 16 | 0.320 | 0.5600 | 0.0833 | 0.0625 | 1.93 |
| `allTiresService` | lgbm | lgbm | 10 | 0.200 | 0.5400 | 0.3030 | 0.5000 | 1.82 |
| `cleaning` | lgbm | lgbm | 7 | 0.140 | 0.8400 | 0.2000 | 0.1429 | 0.09 |
| `paintingPreparation` | lgbm | ridge | 14 | 0.280 | 0.6600 | 0.3704 | 0.3571 | 2.07 |
| `paintingSpraying` | lgbm | lgbm | 11 | 0.220 | 0.6000 | 0.0909 | 0.0909 | 1.96 |
| `paintingFinish` | lgbm | ridge | 6 | 0.120 | 0.7400 | 0.1333 | 0.1667 | 1.16 |
| `hailrepair` | lgbm | ridge+wins(cap=24) | 22 | 0.440 | 0.9600 | 0.9524 | 0.9091 | 2.36 |
| `glas` | lgbm | lgbm | 7 | 0.140 | 0.5200 | 0.2500 | 0.5714 | 2.29 |
| **MACRO** | | | | | **0.6543** | **0.2959** | **0.4048** | **1.72** |
| **FREQ-WT** | | | | | **0.6657** | **0.358** | **0.4444** | **1.89** |
