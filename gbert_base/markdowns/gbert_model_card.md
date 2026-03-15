# Model Card — gbert-base Two-Stage Repair Order Predictor

## Model Overview

| Field | Value |
|---|---|
| **Architecture** | Two-stage: binary classifier (Stage 1) + conditional regressor (Stage 2) per work step |
| **Text encoder** | [`deepset/gbert-base`](https://huggingface.co/deepset/gbert-base) — 768-dim, L2-normalised, frozen |
| **Features** | 768-dim BERT embedding + numeric/keyword/make features |
| **Classifiers** | Logistic Regression + LightGBM (best selected per target by val F1) |
| **Regressors** | Ridge (α=10) + LightGBM (min_child_samples adaptive), best selected per target by val MAE |
| **Targets** | 14 binary work-step labels + 13 duration regressors (hrs) |
| **Data** | `orders_simplified_sample.json` — 491 orders, 60/20/20 split |

---

## Why gbert-base?

TF-IDF treats each token independently. `deepset/gbert-base` (BERT, 110M params, trained on German Wikipedia + OpenLegalData + news) produces **contextual embeddings** — the same word gets different representations depending on context:

> `"Scheibe einbauen"` (install window) vs `"Scheibe ausmessen"` (measure pane)  
> → Different BERT vectors, same TF-IDF term overlap.

This matters most for rare targets (`bodymeasurement`, `allTiresService`) where bag-of-words cannot distinguish subtle positional language.

---

## Feature Engineering

### Text features (replaces TF-IDF)
- Each order's position texts are concatenated (lower-cased, high-value positions only)
- Encoded by `SentenceTransformer("deepset/gbert-base")` with mean-pooling + L2 normalisation
- Output: **768-dimensional dense float32 vector**

### Numeric features (unchanged from baseline)
- Aggregate time/price per cost center (bodywork, painting, hail, …)
- Ratios (painting_time / total_time, etc.)
- Boolean cost-center indicators
- 17 keyword binary flags (German automotive domain regex)
- Car make one-hot encoding (8 top makes + "other" + frequency)

---

## Training Strategy

### Stage 1 — Classification
- One binary classifier per target
- Class weights set automatically based on class imbalance
- Decision threshold tuned per target on validation set (maximise F1, sweep 0.05–0.95)

### Stage 2 — Regression
- Trained **only on positive examples** (duration > 0) per target
- Prevents mean-towards-zero bias from all-orders training
- `mean_fallback` used if < 5 positive training examples

---

## Inference Pipeline

```
Raw order JSON
    │
    ▼
preprocess_positions()       ← clean positions, filter empties
    │
    ├── build_order_text()   ← concatenated position text
    │       │
    │       ▼
    │   SentenceTransformer.encode()  ← 768-dim embedding (per request)
    │
    └── build_numeric_features()  ← numeric/keyword/make vector
    │
    ▼
[BERT embedding | numeric features]   ← concatenated feature vector
    │
    ├── Stage 1 Classifier  →  active? (yes/no)
    │
    └── Stage 2 Regressor   →  duration (hrs)  [only if active]
```

---

## Limitations

| Limitation | Impact |
|---|---|
| **Frozen embeddings** | No domain adaptation; body-shop-specific jargon not in pre-training data is suboptimally encoded |
| **491 training orders** | Rare classes still underrepresented even with BERT |
| **CPU inference** | Each embed call takes ~50–200 ms on CPU — acceptable for single orders, slow for large batches without GPU |
| **No fine-tuning** | With ≥ 5,000 orders, fine-tuning gbert-base end-to-end would significantly improve rare-class recall |

---

## Comparison to TF-IDF Baseline

| Metric | TF-IDF baseline | gbert-base (target) |
|---|---|---|
| Macro F1 (test) | 0.703 | Run `model_gbert.py` to measure |
| Macro Recall (test) | 0.779 | — |
| `bodymeasurement` Recall | 0.25 | Expected improvement |
| `allTiresService` Recall | 0.40 | Expected improvement |

*Actual improvement depends on data. With only 491 orders, gains may be modest.*
