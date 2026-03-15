# Repair Order Prediction Assessment

This repository contains the code, models, and evaluation reports for the **Multi-Target Repair Order Prediction** technical assessment.

## ⏱️ Assessment Timeline & Scope

> **Important Note for Reviewers**
>
> **1. Core Assessment (Within the 4-hour limit)**
> The entirety of the core pipeline—data loading, EDA, feature engineering (numeric + TF-IDF), Train/Val/Test stratification, Stage 1 classifiers (Logistic Regression & LightGBM) with threshold tuning, Stage 2 regressors, and the end-to-end evaluation pipeline—was designed, implemented, and uploaded strictly **within the 4-hour assessment timeline**. 
> - `code/eda_phase1.py`
> - `code/model_phase2.py`
> - `code/predict.py`
> - `markdowns/assessment_writeup.md`
>
> **2. Post-Submission Optimisations (Outside the limit)**
> Following the submission, I performed additional deep-dive analyses to address the asymmetric business cost of false negatives (under-quoting) and the performance of rare minority classes. These experiments and their reports were committed **after** the 4-hour window closed to demonstrate how I approach model iteration and optimization in a real-world setting:
> - `code/model_optimise.py` (Soft-vote ensemble & Recall-constrained tuning)
> - `code/model_oversample.py` (Random oversampling for minority classes)
> - `markdowns/logreg_classifier_metrics.md`
> - `markdowns/oversampling_results.md`
> - `markdowns/optimisation_summary_report.md`

---

## 🏗️ Project Architecture

The solution uses a **Two-Stage Pipeline** to predict 14 distinct work steps (e.g., `bodyrepair`, `paintingSpraying`) and their corresponding durations in **hours** from unstructured JSON order data.

> **Unit note:** Input position-level `totalTime` values in the raw JSON are in **minutes**. The output target values (per work step duration) are aggregated and stored in **hours**. All Stage 2 regression predictions are therefore in hours.

### 1. Feature Engineering
- **Text Features**: Word n-grams (1-2) and Character n-grams (3-5) extracted from the concatenated `calculatedPositions` text via `TfidfVectorizer`.
- **Numeric Features**: Aggregations of time, price, and counts per `genericCostCenter`.
- **Domain Keywords**: 17 hand-crafted regex binary flags (e.g., for ADAS calibration, glass repair) to capture strong domain signals.

### 2. Stage 1 — Multi-Label Classifiers (Occurrence)
Predicts binary occurrence (0 or 1) for each of the 14 targets.
- **Models**: Independent `LogisticRegression` and `LGBMClassifier` models per target.
- **Tuning**: Decision thresholds are swept on the Validation set to maximize F1, rather than using a default 0.5 boundary.

### 3. Stage 2 — Conditional Regressors (Duration)
Predicts the duration (in **hours**) for each target, *conditional* on that target being predicted as active by Stage 1.
- **Models**: `Ridge` Regression and `LGBMRegressor`, trained exclusively on the positive subset for each target.

---

## 🚀 How to Run

1. **Install Dependencies**
   ```bash
   pip install pandas numpy scikit-learn scipy lightgbm matplotlib seaborn
   ```

2. **Run Exploratory Data Analysis (Phase 1)**
   ```bash
   python code/eda_phase1.py
   ```
   *Outputs EDA plots to `eda_plots/` and saves cleaned records to `data/orders_cleaned.json`.*

3. **Run the Core Training Pipeline (Phase 2)**
   ```bash
   python code/model_phase2.py
   ```
   *Trains all classifiers and regressors, evaluates on the test set, saves plots to `model_plots/`, and outputs the final artifact `models/two_stage_pipeline.pkl`.*

4. **Run Post-Submission Optimisations**
   ```bash
   python code/model_optimise.py
   python code/model_oversample.py
   ```
   *Runs ensemble tuning, recall-constrained thresholding, and minority class oversampling experiments using the saved pipeline.*

---

## 📊 Evaluation & Reports

All technical reasoning, baseline comparisons, and business-context explanations are fully documented in the `markdowns/` directory. Start with the core writeup:

- [Assessment Writeup (`assessment_writeup.md`)](markdowns/assessment_writeup.md) — Architecture, rule-based baseline comparison, and final results.
- [Optimisation Summary (`optimisation_summary_report.md`)](markdowns/optimisation_summary_report.md) — Post-submission experiments on soft-voting, recall-constraints, and SMOTE/Oversampling.

---

## 🧠 Experiment: German BERT Feature Upgrade (`gbert_base/`)

A follow-up experiment replacing the TF-IDF bag-of-words text representation with frozen contextual embeddings from [`deepset/gbert-base`](https://huggingface.co/deepset/gbert-base) — a German BERT model (110M params). The rest of the two-stage architecture is unchanged.

| Document | Description |
|---|---|
| [gbert_base/README.md](gbert_base/README.md) | Quickstart, folder structure, and motivation |
| [gbert_evaluation_results.md](gbert_base/markdowns/gbert_evaluation_results.md) | Full results: per-target metrics, macro averages, interpretation, recommendations |
| [gbert_model_card.md](gbert_base/markdowns/gbert_model_card.md) | Model card: architecture, features, inference pipeline, limitations |
| [gbert_setup_guide.md](gbert_base/markdowns/gbert_setup_guide.md) | Setup instructions, run commands, troubleshooting |

**Key result:** Macro Recall improves from **0.779 → 0.781** (+0.2 pp) on the test set. Gains are modest but consistent — expected for frozen embeddings at this data scale (491 orders). Full fine-tuning is recommended once ≥ 3,000 labelled orders are available.
