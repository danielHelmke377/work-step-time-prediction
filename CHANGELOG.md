# Changelog

All notable changes to this project are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.3.0] — 2026-03-15 — Repo Professionalisation

### Added
- `requirements.txt` with pinned dependency ranges for the core pipeline and gbert_base experiment
- Comprehensive `.gitignore` (`.venv/`, `*.pkl`, `model_plots/`, `data/`, OS and IDE files)
- `tests/test_smoke.py` — pipeline loading and embedding shape smoke tests
- `CHANGELOG.md` (this file) + retroactive git tags `v1.0.0`, `v1.1.0`, `v1.2.0`
- `src/repair_order/` package — shared config, feature engineering, and pipeline utilities
- English translations of German-language markdown docs (`markdowns/de/` for originals)
- Module docstring for `code/model_oversample_recall.py`

### Changed
- Replaced all hardcoded `C:\Users\Administrator\baum\...` paths in `code/model_phase2.py`, `code/predict.py`, and `code/eda_phase1.py` with `Path(__file__)`-relative paths

---

## [1.2.0] — 2026-03-15 — German BERT Feature Upgrade

### Added
- `gbert_base/` — full experiment folder replacing TF-IDF with frozen [`deepset/gbert-base`](https://huggingface.co/deepset/gbert-base) embeddings (768-dim, mean-pooled, L2-normalised)
- `gbert_base/code/extract_bert_embeddings.py` — offline embedding extraction via `BertModel`
- `gbert_base/code/model_gbert.py` — two-stage pipeline using BERT features
- `gbert_base/code/predict_gbert.py` — BERT inference script
- `gbert_base/markdowns/gbert_evaluation_results.md` — full evaluation vs TF-IDF baseline
- `gbert_base/markdowns/gbert_model_card.md` — model card
- `gbert_base/markdowns/gbert_setup_guide.md` — setup guide

### Results
- Macro Recall: **0.779 → 0.781** (+0.2 pp) on the test set
- `assembly` Recall: **0.857 → 0.912** (+5.5 pp), largest individual gain

---

## [1.1.0] — 2026-03-14 — Post-Submission Optimisations

### Added
- `code/model_optimise.py` — soft-vote ensemble + recall-constrained threshold tuning
- `code/model_oversample.py` — random oversampling for minority classes (SMOTE-style)
- `code/model_oversample_recall.py` — oversampling combined with recall-floor constraints
- `markdowns/optimisation_summary_report.md` — results and analysis

### Results
- Soft-vote ensemble improved minority class recall on `allTiresService` and `bodymeasurement`
- Recall-constrained thresholding reduced under-quoting risk by catching more true positives

---

## [1.0.0] — 2026-03-13 — Core Assessment Submission

### Added
- `code/eda_phase1.py` — full EDA: sparsity, distributions, co-occurrence, rule baseline
- `code/model_phase2.py` — two-stage pipeline: 14 classifiers + 13 regressors + evaluation
- `code/predict.py` — inference script with explanation and batch mode
- `markdowns/assessment_writeup.md` — architecture, baseline comparison, final results
- Two-stage architecture: `LogisticRegression` + `LGBMClassifier` (Stage 1), `Ridge` + `LGBMRegressor` (Stage 2)
- TF-IDF word/char n-grams + 17 keyword flags + numeric/make features
- Threshold tuning per target on validation set (maximise F1)
- Macro F1 (test): **0.703** | Macro Recall: **0.779** vs rule baseline F1: **0.736**
