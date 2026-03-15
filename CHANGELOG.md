# Changelog

All notable changes to this project are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.5.0] — 2026-03-15 — Combined Best Pipeline

### Added
- `combined_best/` *(local only)* — combined classifier + regressor experiment
  - `combined_best/code/model_combined_best.py` — full pipeline: oversampled soft-vote
    classifiers + F1-opt/recall-floor thresholds + multi-target winsorised regressors
  - `combined_best/markdowns/combined_best_results.md`
- Distribution analysis across all 14 regression targets (skewness, max/p95 ratios)
- Added `combined_best` experiment section to `README.md`

### Changed
- Extended winsorising to 4 high-skew targets (max/p95 ≥ 3): `hailrepair`, `assembly`,
  `calibration`, `paintingFinish`
- Introduced frequency-weighted F1 and Recall as primary evaluation metrics

### Results
- Freq-weighted F1 (F1-opt): **0.9283** | Freq-weighted Recall (RC): **0.9491**
- Freq-weighted MAE: **1.87 hrs** (vs 2.78 hrs baseline, −32.7%)
- Macro F1 (unweighted): 0.8153 — drops vs baseline due to oversampling adding false
  positives on rare targets; freq-weighted view is more representative of real usage

---

## [1.4.0] — 2026-03-15 — hailrepair Regression Experiments

### Added
- `log_transform/` *(local only)* — log1p transform experiment for `hailrepair` regressor
  - `log_transform/code/model_log_transform.py`
  - `log_transform/markdowns/log_transform_results.md`
- `hailrepair_mae_exp/` *(local only)* — three-strategy MAE reduction experiment for `hailrepair`
  - `hailrepair_mae_exp/code/model_mae_experiment.py` — tests 6 variants (baseline, A, B, C, A+C, A+B+C)
  - `hailrepair_mae_exp/markdowns/mae_experiment_results.md`
- Added two experiment sections to `README.md`

### Changed
- `README.md` — updated results section and added experiment summaries
- Corrected MAE unit label in `model_phase2.py` plot (minutes → hours)

### Results — Log-Transform (negative)
- `np.log1p()` transform degraded `hailrepair` MAE: **39.60 → 48.05 hrs (+21.3%)**
- Root cause: `expm1()` amplifies log-space errors exponentially; insufficient at n=29

### Results — MAE Reduction Experiment (positive)
- Best variant: **winsorise at 95th percentile** (Strategy C)
- `hailrepair` MAE: **39.60 → 20.51 hrs (−48.2%)**
- Mean MAE all 14 targets: **3.51 → 2.15 hrs (−38.7%)**
- F1 scores unchanged (classifiers not modified)
- Root cause of baseline error: single 4222-hr training record (likely data error) dominated fit

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
