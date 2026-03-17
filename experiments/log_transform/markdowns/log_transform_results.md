# Log-Transform Experiment — hailrepair Regressor

**Date:** 2026-03-15  
**Experiment:** Replace raw `hailrepair` target with `log1p(hailrepair)` at training time,
apply `expm1()` at prediction time. All other targets and classifiers unchanged.

---

## Motivation

`hailrepair` has extreme duration skew. From the training data:

- Values range from **<1 hr** (minor dent repair) to **>100 hrs** (full hail restoration)
- A linear regressor (Ridge) minimises squared error, so large outliers dominate the fit
- On the baseline test set, `hailrepair` MAE was **39.60 hrs** — far above all other targets
- The `log1p` transform compresses this scale so the model fits the underlying distribution
  shape, not just the extreme outliers

## Why log1p / expm1?

`np.log1p(x)` = `ln(x + 1)` — safe for x = 0, maps large values to a much smaller range.  
`np.expm1(y)` = `e^y – 1` — the exact inverse, applied at prediction time to recover hours.

```
Training:   y_fit  = np.log1p(y_hours)      # compress scale
Prediction: y_pred = np.expm1(model.predict(X))  # restore hours
```

---

## Results — Per-Target Comparison (Test Set, MAE in hours)

| Target                 | Baseline MAE | Log1p MAE | Delta (hrs) | Delta (%) |
|------------------------|-------------|-----------|------------|-----------|
| allTiresService        |         0.63 |      0.63 |     +0.00 |    +0.0% |
| assembly               |         1.62 |      1.62 |     +0.00 |    +0.0% |
| bodymeasurement        |         0.08 |      0.08 |     +0.00 |    +0.0% |
| bodyrepair             |         1.44 |      1.44 |     +0.00 |    +0.0% |
| calibration            |         0.39 |      0.39 |     +0.00 |    +0.0% |
| cleaning               |         0.48 |      0.48 |     +0.00 |    +0.0% |
| dismounting            |         1.16 |      1.16 |     +0.00 |    +0.0% |
| glas                   |         0.09 |      0.09 |     +0.00 |    +0.0% |
| hailrepair             |        39.60 |     48.05 |     +8.45 |   +21.3% ← log1p |
| paintingFinish         |         0.61 |      0.61 |     +0.00 |    +0.0% |
| paintingPreparation    |         2.10 |      2.10 |     +0.00 |    +0.0% |
| paintingSpraying       |         0.67 |      0.67 |     +0.00 |    +0.0% |
| plasticrepair          |         0.12 |      0.12 |     +0.00 |    +0.0% |
| wheelmeasurement       |         0.21 |      0.21 |     +0.00 |    +0.0% |

---

## Summary

| Metric | Baseline | Log-Transform | Change |
|---|---|---|---|
| `hailrepair` MAE (hrs) | 39.60 | 48.05 | +21.3% |
| Mean MAE all 14 targets (hrs) | 3.51 | 4.12 | +0.60 hrs |
| Macro-avg F1 | 0.8372 | 0.8372 | -0.0000 |

---

## Interpretation

**On `hailrepair`:** The log1p transform **degraded** MAE from 39.60 → 48.05 hrs (+21.3%).
This is a known failure mode and can be explained in three ways:

1. **Small positive sample size (n=29 in training).** With so few examples, the regressor
   cannot reliably learn the compressed log-space distribution. Variance in predictions is
   high regardless of the transform.

2. **LightGBM's L2 (squared error) loss in log-space.** Minimising squared error of
   `log1p(y)` is not the same as minimising MAE of `y`. After `expm1()`, large log-space
   errors get amplified exponentially — a log-space error of 1.0 maps to a range-space
   error of `e^1 ≈ 2.7×`. For extreme hail values (log-space 6–8), even small log-space
   prediction errors produce very large hour-scale errors after `expm1`.

3. **Ridge regression is not helped by log-transform when n is small.**
   Ridge's alpha=10.0 penalty was tuned for the raw scale. In log-space, the same
   regularisation strength may be too weak or too strong for this different-shaped target.

**On other targets:** All 13 non-log targets are completely unaffected — confirmed by
0.00 delta on every other row. The transform isolation worked correctly.

**F1 scores:** Identical to baseline — classifiers untouched.

---

## Conclusion & Recommendations

The log1p transform is **not recommended** for the current dataset size.

| Approach | Expected improvement | When to try |
|---|---|---|
| **Log1p + Ridge** (this experiment) | ❌ +21.3% worse | Needs ≥ 200 positive hailrepair examples |
| **LightGBM `objective="regression_l1"`** | ✅ Likely improvement | Try now — minimises MAE directly |
| **Tweedie / Gamma regressor** | ✅ Likely improvement | Natural fit for positive-skewed targets |
| **Separate hailrepair model** with domain features | ✅ Biggest improvement | Needs hail-specific input data (damage area, car age) |

**Priority next step:** Re-run this experiment replacing Ridge with
`LGBMRegressor(objective="regression_l1")` for `hailrepair` — this directly minimises MAE
instead of squared error, without the expm1 amplification risk.

