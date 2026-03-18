# Project Evolution: Work Step Time Prediction

This document details the development history of the Work Step Time Prediction model, from its initial genesis as a 4-hour assessment prototype to its optimized final state.

## 🏆 Assessment vs. Final Optimized State

To provide a clear distinction between what was completed during the timed assessment versus subsequent iterative research, the following table compares the initial baseline against the current architecture:

| Component / Metric | 4-Hour Assessment Baseline | Optimized Final Pipeline |
|---|---|---|
| **Classifier Architecture** | Logistic Regression & LightGBM | LightGBM (Uniform) |
| **Regressor Architecture** | Ridge Regressor (Uniform) | Ridge / LightGBM (Selected per-target) |
| **Decision Thresholds** | Default 0.50 cutoff | F1-Optimized thresholds (0.12 - 0.45) |
| **Outlier Handling** | None | Winsorized at 95th Percentile |
| **Sparse Target Fallback**| Mean duration fill | Median duration fill |
| **Frequency-Weighted F1** | 0.938 | **0.935**  |
| **Frequency-Weighted MAE**| 2.78 hrs | **0.96 hrs** *(65% reduction in time estimation error)* |

*Note: Due to the extremely small dataset size (~500 records), all metrics reported in this repository are based on a single strict internal Train (60%) / Validation (20%) / Test (20%) split. Cross-validation was not utilized in this prototype evaluation.*

*Note: The pipeline can be fully reproduced on synthetic data provided in `data/synthetic_orders.json`. The original proprietary dataset is not included due to NDA restrictions.*

---

## ⏱️ Assessment Timeline & Scope

> **Important Note**
>
> **1. Core Assessment (Within the 4-hour limit)**
> The entirety of the initial core pipeline — data loading, EDA, feature engineering (numeric + TF-IDF), Train/Val/Test stratification, Stage 1 classifiers (Logistic Regression & LightGBM) with threshold tuning, Stage 2 regressors, and the end-to-end evaluation pipeline — was designed, implemented, and uploaded strictly **within the 4-hour assessment timeline**.
> 
> **2. Post-Submission Optimisations (Outside the limit)**
> Following the submission, additional deep-dive analyses were performed to address the asymmetric business cost of false negatives (under-quoting) and the performance of rare minority classes. These were committed **after** the 4-hour window closed to demonstrate real-world model iteration.

---

## 🧪 Post-Submission Optimizations

*The following experiments and iterative improvements were developed post-submission to push the architecture to a more deployment-minded, production-oriented state.*

---

## 🧠 Experiment: German BERT Feature Upgrade (`experiments/gbert_base/`)

A follow-up experiment replacing TF-IDF with frozen contextual embeddings from [`deepset/gbert-base`](https://huggingface.co/deepset/gbert-base) — a German BERT model (110M params). The two-stage architecture is otherwise unchanged.

| Document | Description |
|---|---|
| [gbert_base/README.md](../experiments/gbert_base/README.md) | Quickstart, folder structure, motivation |
| [gbert_evaluation_results.md](../experiments/gbert_base/markdowns/gbert_evaluation_results.md) | Full results: per-target metrics, macro averages, recommendations |
| [gbert_model_card.md](../experiments/gbert_base/markdowns/gbert_model_card.md) | Architecture, features, inference pipeline, limitations |
| [gbert_setup_guide.md](../experiments/gbert_base/markdowns/gbert_setup_guide.md) | Setup instructions, run commands, troubleshooting |

**Key result:** Macro Recall improves from **0.779 → 0.781** (+0.2 pp). Gains are modest but consistent — expected for frozen embeddings at this data scale. Full fine-tuning is recommended once ≥ 3,000 labelled orders are available.

---

## 🔬 Experiment: Log-Transform for `hailrepair` (`experiments/log_transform/`)

Hypothesis: applying `np.log1p()` to the `hailrepair` regression target at training time and `np.expm1()` at prediction time would compress its extreme skew and reduce MAE.

**Result: negative.** MAE increased from **39.60 → 48.05 hrs (+21.3%)**.

Root cause: with only **29 positive training examples**, the `expm1()` inverse transform amplifies log-space prediction errors exponentially. A log-space error of 1.0 maps to a ~2.7× larger error in hours — outweighing any benefit of scale compression.

| Metric | Baseline | Log-Transform |
|---|---|---|
| `hailrepair` MAE | 39.60 hrs | 48.05 hrs ❌ |
| Mean MAE (14 targets) | 3.51 hrs | 4.12 hrs |

---

## 🔬 Experiment: `hailrepair` MAE Reduction (`experiments/hailrepair_mae_exp/`)

Three targeted strategies were tested on the heavily skewed `hailrepair` regressor across 6 variants (individually and combined):

| Strategy | Description |
|---|---|
| **A** | `LGBMRegressor(objective="regression_l1")` — optimises MAE directly |
| **B** | Median fallback instead of mean when n_pos < 5 |
| **C** | Winsorise training targets at 95th percentile |

**Key finding:** winsorising alone (**C**) is the most effective. The training data contains a single 4222-hr record (likely a data error) that dominates the entire fit. Removing it via the 95th-percentile cap halves the error:

| Variant | `hailrepair` MAE | Mean MAE (14) | **Freq-weighted MAE** |
|---|---|---|---|
| Baseline | 39.60 hrs | 3.51 hrs | 2.78 hrs |
| A only (MAE objective) | 27.11 hrs | 2.62 hrs | — |
| **C only (winsorise)** | **20.51 hrs** ✅ | **2.15 hrs** | **1.90 hrs** |
| A + C | 25.17 hrs | 2.48 hrs | — |

F1 scores are **unchanged** across all variants — classifiers are not affected.

**On metrics:** Frequency-weighted MAE (weighted by how often each target appears in real orders) is the most business-relevant metric — it drops to **1.90 hrs** with C_only.

---

## 🔬 Experiment: Combined Best Pipeline (`experiments/combined_best/`)

This experiment explores combining the best classifier and regressor strategies independently. Since Stage 1 (classifiers) and Stage 2 (regressors) are trained on separate subsets with separate objectives, they compose freely.

**Regressor strategy:** winsorise training targets at the 95th percentile for all high-skew targets (max/p95 ≥ 3). Targets and their caps:

| Target | p95 cap | max before | skew |
|---|---|---|---|
| `hailrepair` | 1702.7 hrs | 4222.5 hrs | 2.85 |
| `assembly` | 9.1 hrs | 28.7 hrs | 3.54 |
| `paintingFinish` | 3.6 hrs | 14.3 hrs | 3.96 |
| `calibration` | 4.8 hrs | 18.5 hrs | 4.94 |

**Classifier strategies tested:** baseline pickle (frozen, no retraining), oversampled soft-vote (F1-optimal), and oversampled soft-vote (Recall-floor ≥ 90%).

| Metric | Baseline¹ | **True Optimal** ✅ | Oversample CLF + Winsorise REG |
|---|---|---|---|
| Macro F1 | 0.8372 | **0.8372** (identical) | 0.8153 |
| Macro Recall | 0.779¹ | 0.8594¹ | 0.8445 |
| **Freq-weighted F1** | **0.9387** | **0.9347** | 0.9283 |
| **Freq-weighted Recall** | 0.9460 | 0.9412 | 0.9428 |
| Macro MAE | 3.51 hrs | **2.14 hrs** | 2.15 hrs |
| **Freq-weighted MAE** | 2.78 hrs | **1.88 hrs** | 1.87 hrs |

**Key insight:** since classifiers and regressors are independently trained, the optimal setup is **frozen baseline classifiers + winsorised regressors**.

---

## 🔬 Experiment: Best-Per-Target Regressor (`experiments/combined_best/`)

Extends the true optimal setup by using the **best-performing regressor per target** (LightGBM or Ridge) instead of LightGBM uniformly. For `hailrepair`, the choice between plain Ridge and Ridge+winsorise is made automatically by comparing validation MAE.

| Stage | Configuration |
|---|---|
| Classifiers | Saved baseline from pickle — unchanged |
| Regressors | Per-target winner: LightGBM for 8 targets, Ridge for 5, fallback for 1 |
| `hailrepair` | Plain Ridge (val MAE 0.03 hrs vs 79.74 hrs for winsorised) |

| Metric | True Optimal | **Best-per-target REG** | Delta |
|---|---|---|---|
| Macro F1 | 0.8372 | **0.8372** | ±0 |
| Freq-weighted F1 | 0.9347 | **0.9347** | ±0 |
| Freq-weighted Accuracy | — | **0.9433** | — |
| Macro MAE | 2.14 hrs | **0.70 hrs** | −1.44 hrs |
| **Freq-weighted MAE** | 1.88 hrs | **0.96 hrs** | **−0.92 hrs** |

---

## 🔬 Experiment: Final Optimized Pipeline (LightGBM Classifiers)

Extends the best-per-target regressor setup by standardizing on **LightGBM** uniformly for all 14 Stage 1 classifiers. While a fully mixed LogReg/LGBM ensemble technically scored *marginally* higher on some macro metrics, standardizing on LightGBM was chosen because performance is nearly identical on the business-weighted metrics, while architectural complexity is significantly lower. Thresholds are optimized by validation F1, and no oversampling is used.

| Metric | Baseline | Best-per-REG | **Final Pipeline** | Delta vs BestReg |
|---|---|---|---|---|
| Macro F1 | 0.8372 | 0.8372 | **0.8351** | -0.0021 |
| Freq-weighted F1 | 0.9387 | 0.9347 | **0.9350** | +0.0003 |
| Freq-weighted Accuracy | 0.9433 | 0.9433 | **0.9434** | +0.0001 |
| Macro MAE | 3.51 hrs | 0.70 hrs | **0.70 hrs** | ±0 |
| **Freq-weighted MAE** | 2.78 hrs | **0.96 hrs** | **0.96 hrs** | ±0 |

The MAE stays the same (regressor selection is identical). As shown, standardizing on LightGBM produces virtually identical frequency-weighted metrics to a fully mixed approach. Accepting a slight trade-off in absolute macro F1 to drastically reduce system complexity is a deliberate, deployment-minded design choice. This configuration forms the right-side of the comparison table and the basis of the current `scripts/train.py` core pipeline.
