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

The solution uses a **Two-Stage Pipeline** to predict 14 distinct work steps (e.g., `bodyrepair`, `paintingSpraying`) and their corresponding durations in minutes from unstructured JSON order data.

### 1. Feature Engineering
- **Text Features**: Word n-grams (1-2) and Character n-grams (3-5) extracted from the concatenated `calculatedPositions` text via `TfidfVectorizer`.
- **Numeric Features**: Aggregations of time, price, and counts per `genericCostCenter`.
- **Domain Keywords**: 17 hand-crafted regex binary flags (e.g., for ADAS calibration, glass repair) to capture strong domain signals.

### 2. Stage 1 — Multi-Label Classifiers (Occurrence)
Predicts binary occurrence (0 or 1) for each of the 14 targets.
- **Models**: Independent `LogisticRegression` and `LGBMClassifier` models per target.
- **Tuning**: Decision thresholds are swept on the Validation set to maximize F1, rather than using a default 0.5 boundary.

### 3. Stage 2 — Conditional Regressors (Duration)
Predicts the duration (in minutes) for each target, *conditional* on that target being predicted as active by Stage 1.
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
