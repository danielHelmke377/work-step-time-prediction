# gbert_base — German BERT Feature Upgrade

This folder contains an upgraded version of the repair order two-stage prediction pipeline.  
The core change: **TF-IDF bag-of-words features are replaced by contextual sentence embeddings** from [`deepset/gbert-base`](https://huggingface.co/deepset/gbert-base).

Everything else (classifier architecture, regressor architecture, threshold tuning, evaluation) is identical to the original `code/model_phase2.py`.

---

## Folder Structure

```
gbert_base/
├── code/
│   ├── extract_bert_embeddings.py   # Step 1: extract embeddings (run once)
│   ├── model_gbert.py               # Step 2: train & evaluate full pipeline
│   └── predict_gbert.py             # Step 3: inference on new orders
├── models/
│   └── bert_meta.json               # BERT model version pin
├── markdowns/
│   ├── gbert_model_card.md          # Model card
│   └── gbert_setup_guide.md         # Setup & run instructions
├── data/                            # Auto-populated by extract script
│   ├── bert_embeddings.npy
│   └── bert_order_index.json
├── model_plots/                     # Auto-populated by model_gbert.py
└── README.md
```

---

## Quickstart

```bash
# 1. Install dependencies (first time only)
pip install sentence-transformers torch lightgbm scikit-learn pandas numpy matplotlib seaborn

# 2. Extract BERT embeddings (downloads ~440 MB model on first run)
python gbert_base/code/extract_bert_embeddings.py

# 3. Train pipeline & evaluate
python gbert_base/code/model_gbert.py

# 4. Predict on a demo order
python gbert_base/code/predict_gbert.py --demo 0 --evaluate
```

---

## Why This Approach?

| Feature | TF-IDF (original) | gbert-base (this) |
|---|---|---|
| Vocabulary | Bag-of-words, sparse | Contextual, dense 768-dim |
| German awareness | Partial (substring n-grams) | Native German pre-training |
| Long-range dependencies | ❌ None | ✅ Up to 512 tokens |
| Rare class signal | Weak | Stronger (semantic similarity) |
| Training requirement | None | CPU-only for frozen embeddings |

See `markdowns/gbert_model_card.md` for full details.
