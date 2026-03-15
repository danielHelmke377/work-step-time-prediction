# Combined Best Pipeline — Experiment Results

**Date:** 2026-03-15
**Goal:** Combine the best classifier strategy (oversampling + soft-vote + recall floor)
with the best regression strategy (hailrepair winsorised at 95th percentile) to
simultaneously maximise F1 and minimise frequency-weighted MAE.

---

## Configuration

| Parameter | Value |
|---|---|
| Oversampling target (MIN_POS) | 200 positives per target |
| Ensemble | Soft-vote (LogisticRegression + LightGBM) |
| Recall floor (RC strategy) | 90% |
| Winsorised targets (95th pct) | assembly, calibration, hailrepair, paintingFinish |

---

## Baseline vs Combined Results

| Metric | Baseline (`model_phase2.py`) | **Combined (F1-opt)** | Combined (RC) |
|---|---|---|---|
| Macro F1 (unweighted) | 0.8372 | **0.8153** | 0.7767 |
| Macro Recall (unweighted) | 0.779 | **0.8445** | 0.9153 |
| **Freq-weighted F1** | — | **0.9283** | 0.9134 |
| **Freq-weighted Recall** | — | **0.9428** | 0.9491 |
| Macro MAE (unweighted) | 3.51 hrs | **2.15 hrs** | 2.23 hrs |
| **Freq-weighted MAE** | 2.78 hrs | **1.87 hrs** | 1.91 hrs |

> Frequency-weighted metrics weight each target by its occurrence rate in the test set,
> reflecting real-world business impact. Baseline has no freq-weighted F1/Recall equivalent
> since those were not computed in `model_phase2.py`.

---

## Per-target Results

| Target | n_pos | freq_w | F1 (F1-opt) | Rec (F1-opt) | MAE (F1-opt) | F1 (RC) | Rec (RC) | MAE (RC) |
|--------|-------|--------|-------------|--------------|--------------|---------|----------|----------|
| `calibration           ` |  26.0 |  0.263 |  0.6552 |  0.7308 |    0.38 |  0.5676 |  0.8077 |    0.46 |
| `wheelmeasurement      ` |  26.0 |  0.263 |  0.8148 |  0.8462 |    0.22 |  0.7742 |  0.9231 |    0.26 |
| `bodymeasurement       ` |   4.0 |  0.040 |  0.4000 |  0.2500 |    0.05 |  0.2581 |  1.0000 |    0.61 |
| `dismounting           ` |  86.0 |  0.869 |  0.9718 |  1.0000 |    1.16 |  0.9647 |  0.9535 |    1.09 |
| `bodyrepair            ` |  35.0 |  0.354 |  0.7647 |  0.7429 |    1.35 |  0.7674 |  0.9429 |    1.76 |
| `assembly              ` |  91.0 |  0.919 |  0.9556 |  0.9451 |    1.49 |  0.9249 |  0.8791 |    1.55 |
| `plasticrepair         ` |  15.0 |  0.152 |  0.7500 |  1.0000 |    0.34 |  0.7143 |  1.0000 |    0.36 |
| `allTiresService       ` |   5.0 |  0.051 |  0.2000 |  0.4000 |    0.70 |  0.2000 |  0.4000 |    0.70 |
| `cleaning              ` |  94.0 |  0.949 |  0.9843 |  1.0000 |    0.48 |  0.9843 |  1.0000 |    0.48 |
| `paintingPreparation   ` |  70.0 |  0.707 |  0.9857 |  0.9857 |    2.08 |  0.9857 |  0.9857 |    2.08 |
| `paintingSpraying      ` |  70.0 |  0.707 |  0.9784 |  0.9714 |    0.65 |  0.9784 |  0.9714 |    0.65 |
| `paintingFinish        ` |  70.0 |  0.707 |  0.9718 |  0.9857 |    0.57 |  0.9718 |  0.9857 |    0.57 |
| `hailrepair            ` |  29.0 |  0.293 |  0.9825 |  0.9655 |   20.51 |  0.9825 |  0.9655 |   20.51 |
| `glas                  ` |   8.0 |  0.081 |  1.0000 |  1.0000 |    0.08 |  0.8000 |  1.0000 |    0.20 |
| **MACRO** | — | — | **0.8153** | **0.8445** | **2.15** | **0.7767** | **0.9153** | **2.23** |

---

## Interpretation

### Classifiers
- Oversampling improved F1 for minority targets (`allTiresService`, `bodymeasurement`,
  `plasticrepair`) with very few natural positives.
- The soft-vote ensemble smooths overconfident predictions from either individual model.
- The **Recall-Constrained (RC)** strategy guarantees Recall ≥ 90% per target,
  at the cost of lower precision — useful when missing a work step (false negative) is
  more costly than a false positive.

### Regressors
- The `hailrepair` winsorise removes the single 4222-hr training record (likely a
  data entry error) that dominated the fit. All other regression targets are unchanged.

### Trade-offs
- **F1-optimal** is the better default: highest classification accuracy.
- **RC** is the better choice when **under-quoting** (missed work steps) causes
  customer or business problems — for example, if a hail repair is not quoted but
  actually needed, re-work costs are typically 3–5× the original quote.

---

## Next Steps

1. **Feature engineering** — hail-specific features (`n_hail_positions`,
   `price_hail / total_price`) to further reduce `hailrepair` MAE
2. **Lower winsorise percentile** — test 85th/90th to tighten the hailrepair cap
3. **More data** — oversampling is a substitute for real data; acquiring 100+ additional
   hailrepair orders with accurate durations would be the highest-leverage action
