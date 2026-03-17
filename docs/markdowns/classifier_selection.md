# Classifier Selection — Stage 1 of the Two-Stage Pipeline

## Overview

Stage 1 trains one binary classifier per target to predict whether a work step
occurs in a given repair order. Two algorithms were compared.

---

## 1. Logistic Regression

**What it is:** A linear model that estimates the probability of a work step occurring,
based on a weighted sum of input features.

**Why selected:**
- **Strong with TF-IDF:** Linear models + TF-IDF are a well-established combination for text classification
- **Interpretable:** Feature weights directly show which words are most predictive (e.g. high weight for `frontscheibe` → `glas`)
- **Robust at low data volume:** With only 294 training orders, complex models risk overfitting; Logistic Regression is highly stable
- **Fast:** No tuning required, trains in milliseconds

**Configuration:**
- `C = 1.0` (default regularisation strength)
- Class weights set automatically based on class imbalance per target

---

## 2. LightGBM Classifier

**What it is:** A gradient-boosted decision tree model — learns non-linear patterns and
feature interactions.

**Why selected:**
- **Detects combinations:** e.g. "hail cost-centre AND `DELLEN` keyword AND total_time > 500 min → very likely `hailrepair`" — Logistic Regression cannot learn this directly
- **Tabular data champion:** LightGBM is consistently top-performing on structured/tabular data
- **Handles class imbalance well:** `scale_pos_weight` upweights rare positive classes in the loss function
- **Fast despite complexity:** Significantly faster than Random Forest or XGBoost

**Configuration:**
- `n_estimators = 300`
- `learning_rate = 0.05`
- `num_leaves = 31`
- `scale_pos_weight` computed automatically from class imbalance ratio

---

## Why Not Other Models?

| Model | Reason not tested |
|---|---|
| Random Forest | LightGBM is generally faster and more accurate |
| SVM | Slow on 3,065 sparse features; equivalent to Logistic Regression but slower |
| Neural Networks | 294 training orders is too few — severe overfitting expected |
| Naive Bayes | Suitable for pure text classification but cannot handle numeric features well |

---

## Results: Which Won?

**LightGBM in 11 of 14 cases** — especially for rare or numerically-driven targets:

| Target | Winner | Validation F1 |
|---|---|---|
| `calibration` | LightGBM | 0.790 |
| `wheelmeasurement` | LightGBM | 0.837 |
| `bodymeasurement` | LightGBM | 0.800 |
| `bodyrepair` | LightGBM | 0.875 |
| `assembly` | LightGBM | 0.965 |
| `plasticrepair` | LightGBM | 0.851 |
| `paintingPreparation` | LightGBM | 0.986 |
| `paintingFinish` | LightGBM | 0.986 |
| `hailrepair` | LightGBM | 1.000 |
| `glas` | LightGBM | 0.923 |
| `allTiresService` | LightGBM | 0.000 ⚠️ |
| `dismounting` | Logistic Regression | 0.978 |
| `cleaning` | Logistic Regression | 1.000 |
| `paintingSpraying` | Logistic Regression | 0.986 |

> ⚠️ `allTiresService`: Only 13 positive examples in the full dataset — too rare for reliable machine learning with any model.

---

## Interpretation

This is a typical pattern in mixed-feature pipelines:

- **Clear text signals** (`FAHRZEUGREINIGUNG`, `LACKIERUNG`) → **Logistic Regression is sufficient** — more stable, less prone to overfitting
- **Complex, rare, or numerically-driven targets** (`calibration`, `hailrepair`, `glas`) → **LightGBM wins** — can learn non-linear combinations of text and numeric features

> **Original German version:** [`markdowns/de/classifier_auswahl.md`](de/classifier_auswahl.md)
