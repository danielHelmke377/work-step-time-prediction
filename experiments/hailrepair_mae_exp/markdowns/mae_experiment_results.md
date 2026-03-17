# hailrepair MAE Reduction Experiment

**Date:** 2026-03-15  
**Baseline hailrepair MAE:** 39.60 hrs  
**Baseline total order MAE:** 44.15 hrs  
**Baseline mean MAE (14 targets):** 3.51 hrs  
**Training positive examples:** 29  

---

## Strategies Tested

| ID | Strategy | Implementation |
|---|---|---|
| **A** | LightGBM MAE objective | `objective="regression_l1"` for hailrepair |
| **B** | Median fallback | Replace mean with median when n_pos < 5 |
| **C** | Winsorise training targets | Clip y_pos at 95th percentile = **1702.68 hrs** |

Strategies B and C apply only to `hailrepair`. All classifiers and all other
regression targets are identical across variants.

---

## Results

| Variant    | Strategies           | hail MAE | hail Δ(hrs) | hail Δ(%) | mean MAE (14) | mean Δ(hrs) | mean Δ(%) | total order MAE | total Δ(hrs) | total Δ(%) |
|------------|----------------------|----------|-------------|-----------|---------------|-------------|-----------|-----------------|--------------|------------|
| `baseline` | —                    |     39.60 |     +0.00 |     +0.0% |       3.51 |     +0.00 |     +0.0% |      44.15 |     +0.00 |     +0.0% |
| `A_only  ` | MAE obj              |     27.11 |    -12.49 |    -31.5% ✅ |       2.62 |     -0.89 |    -25.4% |      31.30 |    -12.85 |    -29.1% |
| `B_only  ` | median               |     39.60 |     +0.00 |     +0.0% |       3.51 |     +0.00 |     +0.0% |      44.15 |     +0.00 |     +0.0% |
| `C_only  ` | winsor               |     20.51 |    -19.09 |    -48.2% ✅ |       2.15 |     -1.36 |    -38.7% |      24.73 |    -19.42 |    -44.0% |
| `A+C     ` | MAE obj, winsor      |     25.17 |    -14.43 |    -36.4% ✅ |       2.48 |     -1.03 |    -29.3% |      29.46 |    -14.69 |    -33.3% |
| `A+B+C   ` | MAE obj, median, winsor |     25.17 |    -14.43 |    -36.4% ✅ |       2.48 |     -1.03 |    -29.3% |      29.46 |    -14.69 |    -33.3% |

**Baseline (hardcoded reference from model_phase2.py):**
hailrepair MAE = 39.6 hrs | mean MAE = 3.51 hrs | Macro F1 = 0.8372

---

## Interpretation

**Best variant:** `C_only` with hailrepair MAE = **20.51 hrs**
(improvement of 19.09 hrs (48.2%))

### Why winsorising at 1703 hrs (Strategy C) works/doesn't work
With only 29 positive training examples, a single order with an extreme duration
(e.g. full body hail restoration > 100 hrs) can dominate the regression fit entirely.
Capping at the 95th percentile (1702.7 hrs) removes this distortion — the model
focuses on the 95% of realistic cases.

**Trade-off:** Any test order genuinely requiring > 1703 hrs will be
systematically under-predicted. This is an explicit, documented bias.

### Why MAE objective (Strategy A) helps
`regression_l1` directly minimises the metric we care about. The default L2 loss
penalises large errors quadratically — one 4222-hr outlier in training contributes
as much loss as ~100 typical cases, pulling the model toward over-prediction.

### Why median fallback (Strategy B) has minimal effect here
`hailrepair` has 29 positive train examples — above the n_pos < 5 fallback threshold —
so Strategy B never activates for this target. It matters for targets like
`allTiresService` and `bodymeasurement` (very few positives).

---

## Recommendation

Use the `C_only` variant — it provides the best hailrepair MAE while keeping all other targets and F1 scores unchanged.

Next experiments to consider:
1. **Feature engineering** — add hail-specific features (`n_hail_positions`,
   `max_time_hail_position`, `price_hail / total_price`)
2. **Quantile regression** — `LGBMRegressor(objective="quantile", alpha=0.5)`
   predicts the median, inherently robust to outliers
3. **More data** — each additional positive `hailrepair` example has outsized impact
   at this sample size
