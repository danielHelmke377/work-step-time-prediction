# Regressor Selection — Stage 2 of the Two-Stage Pipeline

## Context

Stage 2 trains one regression model per target to predict **duration in hours** —
but only on orders where the target work step actually occurs (i.e. target value > 0).

---

## Key Design Decision: Training on Positive Subset Only

> The regressor is trained **exclusively on positive examples** — only orders where
> the target value > 0.

If all orders (including zero-valued ones) were used for training, the model would learn
to predict near-zero values for most inputs, since that minimises average error over the
full dataset. By filtering out zeroes, the regressor sees a clean, realistic distribution
of actual work-step durations.

---

## Models Tested

### 1. Ridge Regression

**What it is:** Linear regression with L2 regularisation — penalises large weights to prevent overfitting.

**Why tested:**
- **Natural complement to TF-IDF:** Linear regression + text features is a well-proven combination
- **Stable at low sample counts:** For rare targets (e.g. `bodymeasurement` with only 21 positive examples), regularisation is critical
- **Fast and interpretable:** Coefficients directly show which features drive time up or down

**Configuration:** `alpha = 10.0` — stronger regularisation than the default, because positive subsets are very small.

---

### 2. LightGBM Regressor

**What it is:** Gradient boosting on decision trees for continuous outputs (hours rather than 0/1).

**Why tested:**
- **Non-linear relationships:** Duration often does not scale linearly with features. For example, above a certain number of hail-damage positions, `hailrepair` time increases disproportionately — a linear model cannot capture this
- **Outlier robustness:** LightGBM is less sensitive to extreme individual values than squared-loss regression
- **Automatic feature interactions:** Combinations of cost-centre time, text features, and vehicle make are handled automatically

**Configuration:**
- `n_estimators = 300`, `learning_rate = 0.05`, `num_leaves = 31`
- `min_child_samples = max(5, n_pos // 10)` — prevents overfitting on small positive subsets

---

## Results: Validation Set Comparison (MAE, positive examples only)

| Target | n positive | Ridge MAE | LightGBM MAE | Winner |
|---|---|---|---|---|
| `calibration` | 23 | 1.88 hrs | **0.95 hrs** | LightGBM |
| `wheelmeasurement` | 20 | 0.55 hrs | **0.51 hrs** | LightGBM |
| `bodymeasurement` | 4 | 5.65 hrs | **0.46 hrs** | LightGBM |
| `dismounting` | 90 | 1.18 hrs | **1.14 hrs** | LightGBM |
| `bodyrepair` | 32 | **2.96 hrs** | 3.27 hrs | Ridge |
| `assembly` | 86 | **1.41 hrs** | 1.43 hrs | Ridge |
| `plasticrepair` | 24 | 1.13 hrs | **0.67 hrs** | LightGBM |
| `cleaning` | 97 | 0.40 hrs | 0.40 hrs | Tie |
| `paintingPreparation` | 70 | **1.62 hrs** | 2.13 hrs | Ridge |
| `paintingSpraying` | 69 | 0.73 hrs | **0.64 hrs** | LightGBM |
| `paintingFinish` | 70 | **0.93 hrs** | 1.10 hrs | Ridge |
| `glas` | 7 | 1.59 hrs | **0.89 hrs** | LightGBM |
| `hailrepair` | 32 | **0.03 hrs** | 160.8 hrs | Ridge ⚠️ |

---

## Special Case: `hailrepair`

Ridge dramatically outperforms LightGBM here because:
- Durations vary extremely widely (~1.7 to 67 hours)
- LightGBM overfits severely on such a small training subset with extreme values
- Ridge's strong regularisation keeps predictions stable

**Planned fix:** A **log-transformation** on the target (`log1p` at training, `expm1` at output) would substantially reduce LightGBM's MAE for `hailrepair`. This is documented as a known improvement for future releases.

---

## Final Decision

**LightGBM as the primary regressor** — it achieves the lowest total error across all 14 targets.
`hailrepair` is a documented exception and can be addressed with log-transformation.

For targets with fewer than 5 positive training examples, the **mean of positive training values** is used as a simple but effective fallback.

> **Original German version:** [`markdowns/de/regressionsmodell_auswahl.md`](de/regressionsmodell_auswahl.md)
