# Stage 1 Classifier — Optimisation Results

**Script**: `code/model_optimise.py`  
**Date**: 2026-03-14  
**Two optimisations applied to the trained Stage 1 classifiers:**

1. **Soft-Vote Ensemble** — average LogReg + LightGBM predicted probabilities before threshold application
2. **Recall-Constrained Threshold Tuning** — find threshold that maximises Precision while keeping Recall ≥ 0.9

---

## Optimisation 1 — Soft-Vote Ensemble

### What it does
Instead of choosing *either* Logistic Regression *or* LightGBM (whichever had the higher validation F1), the ensemble averages their probability scores:

```
p_ensemble = (p_logreg + p_lgbm) / 2
```

A new threshold is then tuned on the validation set to maximise F1.

### Validation F1 Comparison (per target)

| Target | F1 LogReg | F1 LightGBM | F1 Ensemble | Best |
|---|:---:|:---:|:---:|:---|
| calibration | 0.6667 | **0.7895** | 0.7500 | lgbm |
| wheelmeasurement | 0.5957 | **0.8372** | 0.7568 | lgbm |
| bodymeasurement | 0.2222 | **0.8000** | 0.4706 | lgbm |
| dismounting | **0.9778** | 0.9609 | 0.9778 | logreg |
| bodyrepair | 0.7719 | 0.8750 | **0.8852** | ensemble |
| assembly | 0.9480 | **0.9651** | 0.9545 | lgbm |
| plasticrepair | 0.5926 | 0.8511 | **0.8750** | ensemble |
| allTiresService | **0.2222** | 0.0000 | 0.2222 | logreg |
| cleaning | **1.0000** | 0.9948 | 1.0000 | logreg |
| paintingPreparation | 0.9714 | **0.9859** | 0.9857 | lgbm |
| paintingSpraying | 0.9855 | 0.9787 | **0.9928** | ensemble |
| paintingFinish | 0.9790 | **0.9859** | 0.9859 | lgbm |
| hailrepair | 0.9846 | **1.0000** | 1.0000 | lgbm |
| glas | 0.4706 | **0.9231** | 0.7143 | lgbm |

### Macro-Average F1 (Validation)

| Model | Macro F1 |
|---|:---:|
| Logistic Regression | 0.7420 |
| LightGBM | 0.8534 |
| **Ensemble** | **0.8265** |

The ensemble is the best single model for **3 out of 14 targets**.

### Ensemble Test Set Results

| Target | Precision | Recall | Accuracy | F1 |
|---|:---:|:---:|:---:|:---:|
| paintingPreparation | 0.9857 | 0.9857 | 0.9798 | 0.9857 |
| cleaning | 0.9691 | 1.0000 | 0.9697 | 0.9843 |
| hailrepair | 1.0000 | 0.9655 | 0.9899 | 0.9825 |
| paintingSpraying | 0.9855 | 0.9714 | 0.9697 | 0.9784 |
| dismounting | 0.9451 | 1.0000 | 0.9495 | 0.9718 |
| paintingFinish | 0.9583 | 0.9857 | 0.9596 | 0.9718 |
| assembly | 0.9663 | 0.9451 | 0.9192 | 0.9556 |
| plasticrepair | 0.7143 | 1.0000 | 0.9394 | 0.8333 |
| wheelmeasurement | 0.8077 | 0.8077 | 0.8990 | 0.8077 |
| bodyrepair | 0.8667 | 0.7429 | 0.8687 | 0.8000 |
| glas | 0.8571 | 0.7500 | 0.9697 | 0.8000 |
| calibration | 0.8235 | 0.5385 | 0.8485 | 0.6512 |
| bodymeasurement | 0.2353 | 1.0000 | 0.8687 | 0.3810 |
| allTiresService | 0.1667 | 0.4000 | 0.8687 | 0.2353 |

**Overall macro-averages (Test Set):**

| Metric | Value |
|---|:---:|
| Macro Precision | 0.8058 |
| Macro Recall | 0.8638 |
| Macro Accuracy | 0.9286 |
| **Macro F1** | **0.8099** |

---

## Optimisation 2 — Recall-Constrained Threshold Tuning

### What it does
In a body-shop context, **missing a work step (false negative) is more costly than
a false alarm (false positive)**: an under-quoted job causes financial loss.

The standard F1-optimal threshold balances precision and recall equally.
This optimisation instead finds, for each target, the **lowest threshold at which
Recall ≥ 0.9** — then picks the one that gives the highest Precision
within that constraint.

### Comparison: F1-Optimal vs Recall-Constrained (Validation, Ensemble probabilities)

| Target | Thr F1 | Prec | Rec | F1 | Thr RC | Prec | Rec | F1 | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| calibration | 0.55 | 0.8824 | 0.6522 | 0.7500 | 0.20 | 0.3889 | 0.9130 | 0.5455 | OK |
| wheelmeasurement | 0.58 | 0.8235 | 0.7000 | 0.7568 | 0.31 | 0.5625 | 0.9000 | 0.6923 | OK |
| bodymeasurement | 0.05 | 0.3077 | 1.0000 | 0.4706 | 0.05 | 0.3077 | 1.0000 | 0.4706 | OK |
| dismounting | 0.10 | 0.9778 | 0.9778 | 0.9778 | 0.54 | 0.9882 | 0.9333 | 0.9600 | OK |
| bodyrepair | 0.52 | 0.9310 | 0.8438 | 0.8852 | 0.07 | 0.5577 | 0.9062 | 0.6905 | OK |
| assembly | 0.51 | 0.9333 | 0.9767 | 0.9545 | 0.95 | 0.9643 | 0.9419 | 0.9529 | OK |
| plasticrepair | 0.48 | 0.8750 | 0.8750 | 0.8750 | 0.41 | 0.7857 | 0.9167 | 0.8462 | OK |
| allTiresService | 0.23 | 0.1250 | 1.0000 | 0.2222 | 0.23 | 0.1250 | 1.0000 | 0.2222 | OK |
| cleaning | 0.05 | 1.0000 | 1.0000 | 1.0000 | 0.02 | 1.0000 | 1.0000 | 1.0000 | OK |
| paintingPreparation | 0.50 | 0.9857 | 0.9857 | 0.9857 | 0.50 | 0.9857 | 0.9857 | 0.9857 | OK |
| paintingSpraying | 0.50 | 0.9857 | 1.0000 | 0.9928 | 0.50 | 0.9857 | 1.0000 | 0.9928 | OK |
| paintingFinish | 0.44 | 0.9722 | 1.0000 | 0.9859 | 0.44 | 0.9722 | 1.0000 | 0.9859 | OK |
| hailrepair | 0.51 | 1.0000 | 1.0000 | 1.0000 | 0.51 | 1.0000 | 1.0000 | 1.0000 | OK |
| glas | 0.50 | 0.7143 | 0.7143 | 0.7143 | 0.34 | 0.4118 | 1.0000 | 0.5833 | OK |

### Macro-Average Summary

| Strategy | Macro Precision | Macro Recall | Macro F1 |
|---|:---:|:---:|:---:|
| F1-Optimal (ensemble) | 0.8224 | 0.9090 | 0.8265 |
| **Recall-Constrained (≥0.9)** | **0.7168** | **0.9641** | **0.7806** |

> **Trade-off**: The recall-constrained strategy raises macro Recall to ≥ 0.96
> at the cost of lower Precision (0.7168 vs 0.8224).
> Use this strategy when under-quoting is unacceptable.

---

## Saved Outputs

| File | Description |
|---|---|
| `model_plots/ensemble_vs_individual.csv` | F1 comparison across all three model types per target |
| `model_plots/ensemble_test_metrics.csv` | Full ensemble test metrics (Precision, Recall, Accuracy, F1) |
| `model_plots/recall_constrained_thresholds.csv` | F1-optimal vs Recall-constrained threshold comparison |

*Generated by `code/model_optimise.py`.*
