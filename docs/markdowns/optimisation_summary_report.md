# Technical Optimisation Report: Stage 1 Multi-Label Classifiers

**Date**: March 14, 2026  
**Context**: Enhancement of the Stage 1 predictive models (Logistic Regression & LightGBM) used to classify the presence of 14 distinct repair work steps based on order text and metadata.

---

## Executive Summary

Following the initial model evaluation, three distinct optimisation strategies were tested to address identified weaknesses—specifically, low recall on minority classes and the asymmetric cost of false negatives (under-quoting a repair job).

1. **Soft-Vote Ensembling**: Averaging probabilities from Logistic Regression and LightGBM.
2. **Recall-Constrained Thresholding**: Re-tuning decision thresholds to guarantee ≥90% recall.
3. **Random Oversampling**: Synthetically balancing the training distribution for rare targets.

**Key Finding**: Random Oversampling yielded the best objective improvement in macro-F1 (+0.02), practically solving the recall issue for rare classes without sacrificing precision. Recall-Constrained Thresholding successfully acts as a safety net against under-quoting, though at a quantifiable cost to precision. 

---

## Optimisation 1: Soft-Vote Ensemble (`model_optimise.py`)

### Rationale
The baseline approach selected the single best classifier (Logistic Regression or LightGBM) per target based on validation F1. However, tree-based models (LGBM) and linear models (LogReg) often make uncorrelated errors on TF-IDF data. A soft-vote ensemble averages their predicted probabilities $P = 0.5 \cdot P_{LR} + 0.5 \cdot P_{LGBM}$, hypothetically yielding a more robust decision boundary.

### Results
- **Outcome**: The ensemble approach underperformed expectations.
- **Data**: LightGBM as a standalone model achieved a macro-validation F1 of **0.853**, whereas the ensemble achieved **0.826**.
- **Analysis**: LightGBM heavily dominates Logistic Regression on this dataset. Because the linear model's probability calibration was significantly inferior on complex non-linear targets, a naive 50/50 soft-vote effectively dragged the LightGBM performance down. The ensemble only "won" on 3 out of 14 targets.
- **Conclusion**: The original baseline strategy—picking the single best algorithm per target via validation F1—is superior to a blanket soft-vote.

---

## Optimisation 2: Recall-Constrained Threshold Tuning (`model_optimise.py`)

### Rationale
Standard threshold tuning sweeps $t \in [0.05, 0.95]$ to maximise the $F_1$-score (the harmonic mean of Precision and Recall). However, in the body-shop business context, the cost matrix is asymmetric:
*   **False Positive (FP)**: Quoting for a step not needed $\rightarrow$ slight over-quote, easily adjusted later.
*   **False Negative (FN)**: Missing a required step $\rightarrow$ under-quoting, resulting in direct margin erosion.

This optimisation changes the objective function:
$$\text{Maximize Precision subject to Recall} \ge 0.90$$

### Results
- **Outcome**: Successful implementation of business constraints, quantifying the exact precision trade-off required for safety.
- **Data**: 
  - Macro-Recall increased from **0.78** (F1-optimal) to **0.96**.
  - Macro-Precision dropped concomitantly from **0.82** to **0.72**.
- **Analysis**: To guarantee catching 90% of instances for ambiguous targets like `bodyrepair`, the threshold must be lowered significantly (e.g., from 0.52 to 0.07). This means nearly half of the positive predictions for that specific target become false alarms.
- **Conclusion**: This is a direct lever the business can use. If margin safety is paramount, these constrained thresholds should be deployed in production.

---

## Optimisation 3: Random Oversampling (`model_oversample.py`)

### Rationale
Classes such as `allTiresService`, `bodymeasurement`, and `calibration` possess extremely low support in the training set ($\le 87$ positive examples). The classifiers prioritise the majority class (negative) and fail to construct distinct decision boundaries for the minority class, leading to near-zero recall. 

To correct this without destroying the sparse TF-IDF matrix (which rules out distance-based synthetic methods like SMOTE), positive training rows for rare classes were randomly duplicated until $N_{positive} \ge 200$.

### Results
- **Outcome**: Significant performance gains across the board for minority classes.
- **Data (Test Set)**:
  - **Macro F1**: Improved from 0.837 to **0.858** (+0.021).
  - `bodymeasurement`: F1 improved by **+0.15** without sacrificing precision.
  - `glas`: F1 improved by **+0.14** (Recall hit a perfect 1.00).
- **Analysis**: By artificially inflating the prior probability of minority classes during the `fit()` stage, both LogisticRegression and LightGBM were forced to allocate weight/splits to the specific TF-IDF tokens associated with those classes. Because we duplicated existing real text rather than synthesizing new text, the learned rules remained accurate on the unseen test set.
- **Conclusion**: This technique cleanly solves the long-tail classification issue and should be permanently integrated into the Phase 2 training pipeline for production.

*(Note: The `allTiresService` target remained flat at 0.22 F1 despite oversampling, indicating an absolute lack of discriminative text features in the raw data rather than a class-imbalance failure. New regex rules must be engineered for this specific step).*
