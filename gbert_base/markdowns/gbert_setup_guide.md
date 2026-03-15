# Setup & Run Guide — gbert_base

## Prerequisites

### Python packages

```bash
pip install sentence-transformers torch lightgbm scikit-learn pandas numpy matplotlib seaborn
```

> **Note on `torch`:** Only CPU is required. `sentence-transformers` will use it automatically.  
> On first run, `deepset/gbert-base` (~440 MB) is downloaded from HuggingFace and cached locally.

---

## Step 1 — Extract BERT Embeddings (run once)

```bash
# From the repo root (c:\Users\Administrator\baum)
python gbert_base/code/extract_bert_embeddings.py
```

**What it does:**
1. Loads all 491 orders from `data/orders_simplified_sample.json`
2. Downloads `deepset/gbert-base` from HuggingFace (cached after first run)
3. Encodes each order's position text as a 768-dim L2-normalised vector
4. Saves output to `gbert_base/data/bert_embeddings.npy` and `bert_order_index.json`

**Expected runtime:** ~2–5 minutes on CPU

**Sanity check printed at end:**
```
  Embedding shape: (491, 768)
  ✓ L2 norm check passed (all rows ≈ 1.0)
```

---

## Step 2 — Train & Evaluate (model_gbert.py)

```bash
python gbert_base/code/model_gbert.py
```

**What it does:**
- Loads pre-computed embeddings from `gbert_base/data/bert_embeddings.npy`
- Builds numeric/keyword/make features (same as baseline)
- Combines BERT (768-dim) + numeric as feature matrix
- Trains Logistic Regression + LightGBM classifiers per target
- Trains Ridge + LightGBM regressors per target (positive subset only)
- Tunes thresholds on validation set
- Evaluates on test set and prints comparison vs TF-IDF baseline
- Saves plots to `gbert_base/model_plots/`
- Saves pipeline to `gbert_base/models/gbert_two_stage_pipeline.pkl`

**Expected runtime:** ~3–8 minutes (LightGBM training on 294 rows is fast)

---

## Step 3 — Inference (predict_gbert.py)

```bash
# Demo from dataset (record index 0), with true labels and keyword explanation
python gbert_base/code/predict_gbert.py --demo 0 --evaluate

# Predict from a JSON file
python gbert_base/code/predict_gbert.py --input path/to/order.json

# Batch of 10 random records
python gbert_base/code/predict_gbert.py --batch 10 --evaluate
```

> **Note:** First call loads `deepset/gbert-base` into memory (~1–2 s).  
> In a production server, load once at startup and reuse for all requests.

---

## Output Files

| File | Description |
|---|---|
| `gbert_base/data/bert_embeddings.npy` | Pre-computed embeddings (491 × 768) |
| `gbert_base/data/bert_order_index.json` | Record index mapping |
| `gbert_base/models/gbert_two_stage_pipeline.pkl` | Trained classifiers + regressors + thresholds |
| `gbert_base/models/bert_meta.json` | BERT model version pin |
| `gbert_base/model_plots/gbert_01_clf_f1_comparison.png` | F1 bar chart per target |
| `gbert_base/model_plots/gbert_02_test_metrics.png` | Test F1 + MAE chart |
| `gbert_base/model_plots/gbert_03_regressor_importance.png` | LightGBM numeric feature importance |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: bert_embeddings.npy` | Run `extract_bert_embeddings.py` first |
| `ModuleNotFoundError: sentence_transformers` | `pip install sentence-transformers` |
| Assertion error: embedding count ≠ record count | Re-run `extract_bert_embeddings.py` after any data change |
| Slow inference | Normal on CPU (~50–200 ms per order). Use GPU for batch production. |
| `Pipeline missing key: make_freq_lookup` | Re-run `model_gbert.py` to regenerate the pickle |
