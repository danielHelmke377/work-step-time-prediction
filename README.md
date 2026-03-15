# Repair Order Prediction

[![CI](https://github.com/danielHelmke377/repair-order-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/danielHelmke377/repair-order-prediction/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predicts **14 binary work steps** (e.g. `bodyrepair`, `paintingSpraying`) and their **duration in hours** from unstructured JSON repair order data, using a two-stage ML pipeline (TF-IDF + LightGBM).

## ⏱️ Assessment Timeline & Scope

> **Important Note for Reviewers**
>
> **1. Core Assessment (Within the 4-hour limit)**
> The entirety of the core pipeline — data loading, EDA, feature engineering (numeric + TF-IDF), Train/Val/Test stratification, Stage 1 classifiers (Logistic Regression & LightGBM) with threshold tuning, Stage 2 regressors, and the end-to-end evaluation pipeline — was designed, implemented, and uploaded strictly **within the 4-hour assessment timeline**.
> - `code/eda_phase1.py`
> - `code/model_phase2.py`
> - `code/predict.py`
> - `markdowns/assessment_writeup.md`
>
> **2. Post-Submission Optimisations (Outside the limit)**
> Following the submission, additional deep-dive analyses were performed to address the asymmetric business cost of false negatives (under-quoting) and the performance of rare minority classes. These were committed **after** the 4-hour window closed to demonstrate real-world model iteration:
> - `code/model_optimise.py` — Soft-vote ensemble & Recall-constrained threshold tuning
> - `code/model_oversample.py` — Random oversampling for minority classes
> - `code/model_oversample_recall.py` — Combined optimisation (oversampling + recall constraints)
> - `markdowns/optimisation_summary_report.md`

---

## 🏗️ Architecture

A **Two-Stage Pipeline** predicts 14 work steps from unstructured JSON order data.

> **Unit note:** Input `totalTime` values in the raw JSON are in **minutes**. All Stage 2 regression outputs are in **hours**.

### Stage 1 — Multi-Label Classifiers (Occurrence)
Predicts binary presence (0/1) for each of the 14 targets independently.
- **Text**: TF-IDF word n-grams (1–2) + character n-grams (3–5) from `calculatedPositions`
- **Numeric**: Time/price aggregations per `genericCostCenter` + 17 domain keyword regex flags
- **Models**: `LogisticRegression` and `LGBMClassifier` per target; thresholds swept on Validation set to maximise F1

### Stage 2 — Conditional Regressors (Duration)
Predicts duration (hours) *only for targets predicted active by Stage 1.*
- **Models**: `Ridge` and `LGBMRegressor`, trained exclusively on the positive subset per target

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/danielHelmke377/repair-order-prediction.git
cd repair-order-prediction

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e ".[dev]"          # installs core deps + pytest
# pip install -e ".[dev,gbert]"  # also installs torch/transformers for gbert_base
```

### 2. Add data

Place `orders_simplified_sample.json` in `data/`.

### 3. Run the pipeline

```bash
# Phase 1 — Exploratory Data Analysis
python code/eda_phase1.py

# Phase 2 — Train all classifiers & regressors
python code/model_phase2.py

# Inference on a single order
python code/predict.py

# Post-submission optimisations (require trained pipeline)
python code/model_optimise.py
python code/model_oversample.py
python code/model_oversample_recall.py
```

### 4. Run tests

```bash
pytest tests/ -v
```

---

## 📁 Repository Structure

```
.
├── code/                        # All runnable scripts
│   ├── eda_phase1.py            # Phase 1: EDA
│   ├── model_phase2.py          # Phase 2: core training pipeline
│   ├── predict.py               # Inference on a single order
│   ├── model_optimise.py        # Soft-vote ensemble + recall-constrained thresholds
│   ├── model_oversample.py      # Random oversampling experiment
│   └── model_oversample_recall.py  # Combined optimisation
│
├── src/repair_order/            # Shared Python package
│   ├── config.py                # Constants (targets, keywords, makes)
│   ├── features.py              # Feature engineering functions
│   └── pipeline.py             # Pipeline load + predict utilities
│
├── tests/                       # Pytest smoke tests
├── markdowns/                   # English documentation & reports
│   ├── de/                      # Original German versions (archived)
│   ├── assessment_writeup.md
│   ├── classifier_selection.md
│   ├── ngrams_explained.md
│   ├── regressor_selection.md
│   └── ...
├── gbert_base/                  # German BERT experiment (see below)
├── models/                      # Saved pipeline artifacts (git-ignored)
├── data/                        # Raw JSON data (git-ignored)
├── CHANGELOG.md
├── pyproject.toml
└── requirements.txt
```

---

## 📊 Results

| Stage | Metric | Value |
|---|---|---|
| Stage 1 (Classifiers) | Macro F1 | **0.914** |
| Stage 1 (Classifiers) | Macro Recall | **0.779** |
| Stage 2 (Regressors) | Macro MAE | **3.5 hrs** (vs 20.5 hrs rule baseline — 6× lower) |

See [`markdowns/assessment_writeup.md`](markdowns/assessment_writeup.md) for full per-target breakdowns.

---

## 🧠 Experiment: German BERT Feature Upgrade (`gbert_base/`)

A follow-up experiment replacing TF-IDF with frozen contextual embeddings from [`deepset/gbert-base`](https://huggingface.co/deepset/gbert-base) — a German BERT model (110M params). The two-stage architecture is otherwise unchanged.

| Document | Description |
|---|---|
| [gbert_base/README.md](gbert_base/README.md) | Quickstart, folder structure, motivation |
| [gbert_evaluation_results.md](gbert_base/markdowns/gbert_evaluation_results.md) | Full results: per-target metrics, macro averages, recommendations |
| [gbert_model_card.md](gbert_base/markdowns/gbert_model_card.md) | Architecture, features, inference pipeline, limitations |
| [gbert_setup_guide.md](gbert_base/markdowns/gbert_setup_guide.md) | Setup instructions, run commands, troubleshooting |

**Key result:** Macro Recall improves from **0.779 → 0.781** (+0.2 pp). Gains are modest but consistent — expected for frozen embeddings at this data scale. Full fine-tuning is recommended once ≥ 3,000 labelled orders are available.

---

## 🔬 Experiment: Log-Transform for `hailrepair` (`log_transform/`)

Hypothesis: applying `np.log1p()` to the `hailrepair` regression target at training time
and `np.expm1()` at prediction time would compress its extreme skew and reduce MAE.

**Result: negative.** MAE increased from **39.60 → 48.05 hrs (+21.3%)**.

Root cause: with only **29 positive training examples**, the `expm1()` inverse transform
amplifies log-space prediction errors exponentially. A log-space error of 1.0 maps to a
~2.7× larger error in hours — outweighing any benefit of scale compression.

| Metric | Baseline | Log-Transform |
|---|---|---|
| `hailrepair` MAE | 39.60 hrs | 48.05 hrs ❌ |
| Mean MAE (14 targets) | 3.51 hrs | 4.12 hrs |

Full analysis: `log_transform/markdowns/log_transform_results.md` *(local only, not pushed)*

---

## 🔬 Experiment: `hailrepair` MAE Reduction (`hailrepair_mae_exp/`)

Three targeted strategies were tested on the heavily skewed `hailrepair` regressor
across 6 variants (individually and combined):

| Strategy | Description |
|---|---|
| **A** | `LGBMRegressor(objective="regression_l1")` — optimises MAE directly |
| **B** | Median fallback instead of mean when n_pos < 5 |
| **C** | Winsorise training targets at 95th percentile |

**Key finding:** winsorising alone (**C**) is the most effective. The training data
contains a single 4222-hr record (likely a data error) that dominates the entire fit.
Removing it via the 95th-percentile cap halves the error:

| Variant | `hailrepair` MAE | Mean MAE (14) | **Freq-weighted MAE** |
|---|---|---|---|
| Baseline | 39.60 hrs | 3.51 hrs | 2.78 hrs |
| A only (MAE objective) | 27.11 hrs | 2.62 hrs | — |
| **C only (winsorise)** | **20.51 hrs** ✅ | **2.15 hrs** | **1.90 hrs** |
| A + C | 25.17 hrs | 2.48 hrs | — |

F1 scores are **unchanged** across all variants — classifiers are not affected.

**On metrics:** Frequency-weighted MAE (weighted by how often each target appears in real
orders) is the most business-relevant metric — it drops to **1.90 hrs** with C_only.
MAPE was also evaluated but is unsuitable here: `bodyrepair` hits MAPE of 6255% due to
near-zero true durations in some test records, making the macro MAPE (~513%) uninformative.

Full analysis: `hailrepair_mae_exp/markdowns/mae_experiment_results.md` *(local only, not pushed)*

---

## 🔬 Experiment: Combined Best Pipeline (`combined_best/`)

This experiment explores combining the best classifier and regressor strategies
independently. Since Stage 1 (classifiers) and Stage 2 (regressors) are trained on
separate subsets with separate objectives, they compose freely.

**Regressor strategy:** winsorise training targets at the 95th percentile for all
high-skew targets (max/p95 ≥ 3). Targets and their caps:

| Target | p95 cap | max before | skew |
|---|---|---|---|
| `hailrepair` | 1702.7 hrs | 4222.5 hrs | 2.85 |
| `assembly` | 9.1 hrs | 28.7 hrs | 3.54 |
| `paintingFinish` | 3.6 hrs | 14.3 hrs | 3.96 |
| `calibration` | 4.8 hrs | 18.5 hrs | 4.94 |

**Classifier strategies tested:** baseline (no oversampling), oversampled soft-vote
(F1-optimal), and oversampled soft-vote (Recall-floor ≥ 90%).

| Metric | Baseline CLF | **Baseline CLF + Winsorise REG** ✅ | Oversample CLF + Winsorise REG |
|---|---|---|---|
| Macro F1 | 0.8372 | 0.8099 | 0.8153 |
| Macro Recall | 0.779 | 0.8637 | 0.8445 |
| **Freq-weighted F1** | **0.9387** | **0.9294** | 0.9283 |
| **Freq-weighted Recall** | 0.9460 | 0.9348 | 0.9428 |
| Macro MAE | 3.51 hrs | 2.14 hrs | 2.15 hrs |
| **Freq-weighted MAE** | 2.78 hrs | **1.87 hrs** | **1.87 hrs** |

*Freq-weighted metrics weight each target by its test-set occurrence rate.*

**Key insight:** since classifiers and regressors are independently trained, the
optimal setup is simply **baseline classifiers + winsorised regressors**:
- The baseline classifiers already achieve the highest freq-weighted F1 (0.9387) — oversampling
  of rare targets adds false positives on the *common* high-frequency targets, reducing their
  precision and pulling freq-weighted F1 down to 0.9283
- The winsorised regressors reduce freq-weighted MAE by **−32.7%** regardless of
  which classifiers are used — the MAE gain comes entirely from the regressor stage

**Recommendation:** apply the 4-target 95th-percentile winsorise to `model_phase2.py`
as a one-parameter change. No classifier retraining needed. The RC (Recall-floor ≥ 90%)
oversampled variant is only worth considering if minimising missed work steps
(under-quoting risk) is more important than precision.

Full analysis: `combined_best/markdowns/combined_best_results.md` *(local only, not pushed)*




