# Optmisation 3 — Random Oversampling for Minority Classes

**Script**: `code/model_oversample.py`  
**Date**: 2026-03-14  

## Rationale
Some work steps are incredibly rare (e.g., `allTiresService`, `bodymeasurement`). The original Logistic Regression & LightGBM classifiers struggled to learn the text boundaries for these classes.

In this experiment, we applied **Random Oversampling** to the training matrix (duplicating positive rows until every class had at least `200` positive examples), retrained both classifiers, and tuned new optimal F1 thresholds. We then compare the best oversampled classifier per-target against the best baseline classifier.

---

## Test Set Results 

*(Comparing the best Baseline classifier vs the best Oversampled classifier per target)*

| Target | Test Pos Count | Baseline Recall | OS Recall | Recall Δ | Baseline Precision | OS Precision | Prec Δ | Baseline F1 | OS F1 | F1 Δ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| calibration | 26 | 0.5000 | 0.5769 | **+0.0769** | 0.9286 | 0.8824 | -0.0462 | 0.6500 | 0.6977 | **+0.0477** |
| wheelmeasurement | 26 | 0.8846 | 0.8846 | +0.0000 | 0.8214 | 0.7931 | -0.0283 | 0.8519 | 0.8364 | -0.0155 |
| bodymeasurement | 4 | 0.7500 | 0.7500 | +0.0000 | 0.5000 | 0.7500 | **+0.2500** | 0.6000 | 0.7500 | **+0.1500** |
| dismounting | 86 | 0.9767 | 0.9767 | +0.0000 | 0.9438 | 0.9655 | **+0.0217** | 0.9600 | 0.9711 | **+0.0111** |
| bodyrepair | 35 | 0.9143 | 0.8571 | -0.0571 | 0.8000 | 0.8108 | **+0.0108** | 0.8533 | 0.8333 | -0.0200 |
| assembly | 91 | 0.9341 | 0.9341 | +0.0000 | 0.9659 | 0.9659 | +0.0000 | 0.9497 | 0.9497 | +0.0000 |
| plasticrepair | 15 | 1.0000 | 1.0000 | +0.0000 | 0.8824 | 0.8333 | -0.0491 | 0.9375 | 0.9091 | -0.0284 |
| allTiresService | 5 | 0.4000 | 0.4000 | +0.0000 | 0.1538 | 0.1538 | +0.0000 | 0.2222 | 0.2222 | +0.0000 |
| cleaning | 94 | 1.0000 | 1.0000 | +0.0000 | 0.9691 | 0.9691 | +0.0000 | 0.9843 | 0.9843 | +0.0000 |
| paintingPreparation | 70 | 1.0000 | 1.0000 | +0.0000 | 0.9722 | 0.9722 | +0.0000 | 0.9859 | 0.9859 | +0.0000 |
| paintingSpraying | 70 | 0.9714 | 0.9714 | +0.0000 | 0.9714 | 0.9855 | **+0.0141** | 0.9714 | 0.9784 | **+0.0070** |
| paintingFinish | 70 | 0.9857 | 0.9857 | +0.0000 | 0.9583 | 0.9583 | +0.0000 | 0.9718 | 0.9718 | +0.0000 |
| hailrepair | 29 | 0.9655 | 0.9655 | +0.0000 | 1.0000 | 1.0000 | +0.0000 | 0.9825 | 0.9825 | +0.0000 |
| glas | 8 | 0.7500 | 1.0000 | **+0.2500** | 0.8571 | 0.8889 | **+0.0318** | 0.8000 | 0.9412 | **+0.1412** |

### Macro-Average Test Set Performance

| Metric | Baseline | Oversampled | Delta |
|---|:---:|:---:|:---:|
| **Macro Precision** | 0.8374 | 0.8521 | +0.0146 |
| **Macro Recall** | 0.8594 | 0.8787 | +0.0193 |
| **Macro F1** | 0.8372 | 0.8581 | +0.0209 |

---

## Conclusion
Random oversampling successfully improved the overall model performance! The macro F1 improved by +0.0209.
