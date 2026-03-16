# Best-Per-Target Regressor — Saved CLF + Mixed REG

**Date:** 2026-03-16

**Goal:** Use the saved baseline classifiers from `models/two_stage_pipeline.pkl`
and train the best-performing regressor per target as documented in
`markdowns/regressor_selection.md`. For `hailrepair`, Ridge vs Ridge+winsorise
is selected automatically based on validation MAE.

## Regressor Assignments

| Target | Regressor | Reason |
|---|---|---|
| `calibration` | LightGBM | Lower val MAE (0.95 vs 1.88 hrs)  |
| `wheelmeasurement` | LightGBM | Lower val MAE (0.51 vs 0.55 hrs) |
| `bodymeasurement` | LightGBM | Lower val MAE (0.46 vs 5.65 hrs) |
| `dismounting` | LightGBM | Lower val MAE (1.14 vs 1.18 hrs) |
| `bodyrepair` | Ridge | Lower val MAE (2.96 vs 3.27 hrs) |
| `assembly` | Ridge | Lower val MAE (1.41 vs 1.43 hrs) |
| `plasticrepair` | LightGBM | Lower val MAE (0.67 vs 1.13 hrs) |
| `cleaning` | LightGBM | Tie — LightGBM primary choice |
| `paintingPreparation` | Ridge | Lower val MAE (1.62 vs 2.13 hrs) |
| `paintingSpraying` | LightGBM | Lower val MAE (0.64 vs 0.73 hrs) |
| `paintingFinish` | Ridge | Lower val MAE (0.93 vs 1.10 hrs) |
| `hailrepair` | ridge | Chosen by val MAE comparison |
| `glas` | LightGBM | Lower val MAE (0.89 vs 1.59 hrs) |
| `allTiresService` | Fallback (mean) | n_pos < 5 |

## Comparison vs True Optimal (all-LightGBM + winsorise)

| Metric | True Optimal | **Best-per-REG** | Delta |
|---|---|---|---|
| Macro F1 | 0.8372 | **0.8372** | +0.0000 |
| Macro Recall | 0.8594 | **0.8594** | +0.0000 |
| Macro Accuracy | — | **0.938** | — |
| Freq-weighted F1 | 0.9347 | **0.9347** | +0.0000 |
| Freq-weighted Recall | 0.9412 | **0.9412** | +0.0000 |
| Freq-weighted Accuracy | — | **0.9433** | — |
| Macro MAE | 2.14 hrs | **0.7 hrs** | -1.44 hrs |
| **Freq-weighted MAE** | 1.88 hrs | **0.96 hrs** | -0.92 hrs |

## Per-target Results

| Target | Regressor | n_pos | freq_w | Accuracy | F1 | Recall | MAE |
|--------|-----------|-------|--------|----------|----|--------|-----|
| `calibration` | lgbm | 26 | 0.263 | 0.8586 | 0.6500 | 0.5000 | 0.39 |
| `wheelmeasurement` | lgbm | 26 | 0.263 | 0.9192 | 0.8519 | 0.8846 | 0.21 |
| `bodymeasurement` | lgbm | 4 | 0.040 | 0.9596 | 0.6000 | 0.7500 | 0.08 |
| `dismounting` | lgbm | 86 | 0.869 | 0.9293 | 0.9600 | 0.9767 | 1.16 |
| `bodyrepair` | ridge | 35 | 0.354 | 0.8889 | 0.8533 | 0.9143 | 1.73 |
| `assembly` | ridge | 91 | 0.919 | 0.9091 | 0.9497 | 0.9341 | 1.59 |
| `plasticrepair` | lgbm | 15 | 0.152 | 0.9798 | 0.9375 | 1.0000 | 0.12 |
| `allTiresService` | lgbm | 5 | 0.051 | 0.8586 | 0.2222 | 0.4000 | 0.63 |
| `cleaning` | lgbm | 94 | 0.949 | 0.9697 | 0.9843 | 1.0000 | 0.48 |
| `paintingPreparation` | ridge | 70 | 0.707 | 0.9798 | 0.9859 | 1.0000 | 1.91 |
| `paintingSpraying` | lgbm | 70 | 0.707 | 0.9596 | 0.9714 | 0.9714 | 0.67 |
| `paintingFinish` | ridge | 70 | 0.707 | 0.9596 | 0.9718 | 0.9857 | 0.77 |
| `hailrepair` | ridge | 29 | 0.293 | 0.9899 | 0.9825 | 0.9655 | 0.02 |
| `glas` | lgbm | 8 | 0.081 | 0.9697 | 0.8000 | 0.7500 | 0.09 |
| **MACRO** | — | — | — | **0.938** | **0.8372** | **0.8594** | **0.7** |
| **FREQ-WT** | — | — | — | **0.9433** | **0.9347** | **0.9412** | **0.96** |
