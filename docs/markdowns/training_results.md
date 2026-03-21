# Two-Stage Pipeline — Training Results

**Date:** 2026-03-21

**Stage 1:** `LGBMClassifier` (one per target, no oversampling).
Decision thresholds F1-tuned on the validation set.

**Stage 2:** Per-target best regressor from `BEST_REG`
(`lgbm`, `ridge`, or `ridge_auto`). Targets with < 5 positive
training samples fall back to the mean of positives.

## Per-target Results

| Target | clf | reg | n_pos | freq_w | Accuracy | F1 | Recall | MAE |
|--------|-----|-----|-------|--------|----------|----|--------|-----|
| `calibration` | lgbm | lgbm | 18 | 0.180 | 0.6000 | 0.2000 | 0.2778 | 1.62 |
| `wheelmeasurement` | lgbm | lgbm | 21 | 0.210 | 0.6300 | 0.2745 | 0.3333 | 0.35 |
| `bodymeasurement` | lgbm | lgbm | 9 | 0.090 | 0.6700 | 0.1081 | 0.2222 | 1.54 |
| `dismounting` | lgbm | lgbm | 11 | 0.110 | 0.6500 | 0.2222 | 0.4545 | 1.65 |
| `bodyrepair` | lgbm | ridge | 19 | 0.190 | 0.6700 | 0.2979 | 0.3684 | 1.58 |
| `assembly` | lgbm | ridge | 16 | 0.160 | 0.5800 | 0.1923 | 0.3125 | 1.89 |
| `plasticrepair` | lgbm | lgbm | 19 | 0.190 | 0.6200 | 0.2963 | 0.4211 | 1.70 |
| `allTiresService` | lgbm | lgbm | 14 | 0.140 | 0.7200 | 0.2222 | 0.2857 | 1.14 |
| `cleaning` | lgbm | lgbm | 21 | 0.210 | 0.7000 | 0.1176 | 0.0952 | 0.22 |
| `paintingPreparation` | lgbm | ridge | 17 | 0.170 | 0.8000 | 0.2308 | 0.1765 | 0.86 |
| `paintingSpraying` | lgbm | lgbm | 17 | 0.170 | 0.6800 | 0.1111 | 0.1176 | 1.31 |
| `paintingFinish` | lgbm | ridge | 15 | 0.150 | 0.8500 | 0.0000 | 0.0000 | 0.52 |
| `hailrepair` | lgbm | ridge | 50 | 0.500 | 0.9200 | 0.9167 | 0.8800 | 2.93 |
| `glas` | lgbm | lgbm | 20 | 0.200 | 0.5300 | 0.2295 | 0.3500 | 2.11 |
| **MACRO** | | | | | **0.6871** | **0.2442** | **0.3068** | **1.39** |
| **FREQ-WT** | | | | | **0.7127** | **0.3332** | **0.3783** | **1.55** |
