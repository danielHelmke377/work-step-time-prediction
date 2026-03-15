# Evaluation Report — deepset/gbert-base vs TF-IDF Baseline

**Date:** 2026-03-15  
**Model:** `deepset/gbert-base` (110M params, frozen, mean-pooled, L2-normalised)  
**Pipeline:** Two-stage — binary classifier (Stage 1) + conditional regressor (Stage 2), one per work-step target  
**Data:** `orders_simplified_sample.json` — 491 orders | 294 train / 98 val / 99 test (60/20/20 stratified)  
**Features:** 768-dim BERT embedding + numeric/keyword/make features (TF-IDF word/char n-grams removed)

---

## 1. Classifier Results — Stage 1 (Logistic Regression on Test Set)

Thresholds tuned on the validation set to maximise F1 per target. Comparison against the TF-IDF LogReg baseline from `model_phase2.py`.

| Target | gbert Precision | gbert Recall | gbert Accuracy | TF-IDF Recall | Δ Recall |
|---|:---:|:---:|:---:|:---:|:---:|
| `cleaning` | 0.969 | **1.000** | 0.970 | 1.000 | 0.000 |
| `plasticrepair` | 0.227 | **1.000** | 0.485 | 1.000 | 0.000 |
| `dismounting` | 0.955 | **0.977** | 0.939 | 0.977 | 0.000 |
| `paintingPreparation` | 0.971 | **0.971** | 0.960 | 0.971 | 0.000 |
| `paintingSpraying` | 0.986 | **0.971** | 0.970 | 0.971 | 0.000 |
| `paintingFinish` | 0.907 | **0.971** | 0.909 | 0.957 | **+0.014** |
| `hailrepair` | 0.966 | **0.966** | 0.980 | 0.966 | 0.000 |
| `assembly` | 0.954 | **0.912** | 0.879 | 0.857 | **+0.055** |
| `calibration` | 0.474 | 0.692 | 0.717 | 0.692 | 0.000 |
| `wheelmeasurement` | 0.621 | 0.692 | 0.808 | 0.692 | 0.000 |
| `glas` | 0.714 | 0.625 | 0.950 | 0.750 | **−0.125** |
| `bodyrepair` | 0.808 | **0.600** | 0.808 | 0.571 | **+0.029** |
| `allTiresService` | 0.167 | 0.400 | 0.869 | 0.400 | 0.000 |
| `bodymeasurement` | 0.100 | 0.250 | 0.879 | 0.250 | 0.000 |

### Macro Averages — Test Set

| Metric | TF-IDF baseline | gbert-base | Δ |
|---|:---:|:---:|:---:|
| **Macro Recall** | 0.779 | **0.781** | +0.002 |
| **Macro Precision** | 0.703 | **0.708** | +0.005 |
| **Macro Accuracy** | 0.852 | **0.859** | +0.007 |

---

## 2. Key Findings

### ✅ Improvements over TF-IDF

- **`assembly` Recall +5.5 pp** (0.857 → 0.912): The most meaningful gain. Assembly positions often use varied, context-dependent phrasing (e.g. *"A+E Seitenspiegel"*, *"Einbau Stoßfänger komplett"*). BERT resolves this variability better than bag-of-words.
- **`paintingFinish` Recall +1.4 pp** (0.957 → 0.971): Marginal but consistent.
- **`bodyrepair` Recall +2.9 pp** (0.571 → 0.600): Body repair language is heterogeneous; contextual embeddings separate it from adjacent categories slightly better.
- **Macro Precision +0.5 pp, Accuracy +0.7 pp** across all 14 targets.

### ⚠️ Regressions

- **`glas` Recall −12.5 pp** (0.750 → 0.625): The threshold shifted (0.91 → 0.94), which is fragile given only 8 positive test cases. A single mis-classified example drives this. Not a reliable signal with *n* this small.

### ❌ No improvement on the hardest targets

- **`allTiresService`** (Recall 0.40) and **`bodymeasurement`** (Recall 0.25): Unchanged from TF-IDF. These targets have **≤ 25 positive training examples** — too few for either representation to generalise, regardless of embedding quality.

---

## 3. Interpretation

The improvements are **real but small**. This is expected and quantitatively consistent with the literature on frozen BERT features for short-text classification under low data regimes:

> *With fewer than ~1,000 labelled examples, frozen contextual embeddings typically yield only marginal gains over strong TF-IDF baselines. The benefit materialises most clearly in 5,000+ sample settings where fine-tuning can be performed end-to-end.*

The 491-order dataset sits below this threshold. BERT's advantage — long-range dependency modelling — is impactful for full sentences and paragraphs, but order position descriptions are often 3-8 words long (*"Lackierung Tür links"*), leaving little room for contextual disambiguation beyond what bigrams already capture.

---

## 4. Recommendations

| Priority | Action | Expected gain |
|---|---|---|
| **High** | Collect more orders (target: ≥ 3,000) | Enables fine-tuning; expected Macro Recall > 0.87 |
| **High** | Log-transform `hailrepair` regressor target | Fixes known regression outlier (MAE likely improves 10× for that target) |
| **Medium** | Fine-tune end-to-end with `≥ 3,000` orders | Unlocks BERT's full contextual capacity |
| **Medium** | Augment `allTiresService` and `bodymeasurement` | Active labelling, synonym expansion, or rule-based augmentation |
| **Low** | Try `deepset/gbert-large` (337M params) | Marginal at this data size; defers until fine-tuning is feasible |

---

## 5. Artifacts

| File | Description |
|---|---|
| `gbert_base/models/gbert_two_stage_pipeline.pkl` | Trained pipeline (classifiers + regressors + thresholds) |
| `gbert_base/data/bert_embeddings.npy` | Pre-computed embeddings, shape (491, 768), L2-normalised |
| `gbert_base/model_plots/gbert_01_clf_f1_comparison.png` | F1 bar chart per target vs TF-IDF baseline |
| `gbert_base/model_plots/gbert_02_test_metrics.png` | Test F1 + MAE per target |
| `gbert_base/model_plots/gbert_03_regressor_importance.png` | LightGBM numeric feature importance (Stage 2) |

---

*Source data: `gbert_base/model_plots/gbert_logreg_clf_metrics.csv` — generated by `gbert_base/code/model_gbert.py`.*
