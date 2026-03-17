# Work Step Time Prediction Pipeline

[![CI](https://github.com/danielHelmke377/work-step-time-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/danielHelmke377/work-step-time-prediction/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predicts **14 binary work steps** (e.g., `bodyrepair`, `paintingSpraying`) and their **duration in hours** from unstructured JSON repair order data. 

This repository represents the **first step in prototyping the best possible model** for a work-step time prediction pipeline, evolving rapidly from a 4-hour assessment into a highly optimized architecture. While not yet a fully integrated production system, it demonstrates the systematic iteration required to build one.

To see how this model evolved from a rapid 4-hour prototype to its highly optimized final state, explore the experimental logs inside the [Project Evolution Summary](docs/project_evolution.md).

## 🚀 Impact & Results

The final pipeline achieves state-of-the-art performance for the business context, optimizing heavily for frequency-weighted metrics (weighting targets by how commonly work steps appear in actual orders).

| Metric | Value |
|---|---|
| **Macro F1** | **0.838** |
| **Frequency-Weighted F1** | **0.935** |
| **Frequency-Weighted MAE** | **0.96 hrs** |
| **Frequency-Weighted Accuracy** | **0.943** |

## 🏗️ Architecture

A **Two-Stage Pipeline** handles the multi-label to regression problem:

1. **Stage 1 — Multi-Label Classifiers (Occurrence):** Predicts binary presence (0/1) for each of the 14 targets independently. Uses a mix of `LogisticRegression` and `LGBMClassifier` (best-per-target selected by validation F1).
   - Features: TF-IDF word n-grams, character n-grams, time/price aggregations per cost-center, and domain keyword regex flags.
2. **Stage 2 — Conditional Regressors (Duration):** Predicts duration (hours) *only for targets predicted active by Stage 1*. Uses a mix of `Ridge` Regression and `LGBMRegressor` depending on the target dataset size and skew.

## 💻 Repository Usage (Portfolio Example)

> [!NOTE]
> **Data Privacy:** Due to customer confidentiality and NDA restrictions, the proprietary JSON repair order dataset used to train this model is **not** included in this public repository. 
> 
> As such, you cannot directly run the training or inference scripts. This repository is made public purely as an **architectural portfolio example**, demonstrating how to structure, test, and relentlessly optimize a machine learning solution for a complex business problem.

### Exploring the Architecture

If you are reviewing this repository:
1. **Start with the [Project Evolution Summary](docs/project_evolution.md)**: This document is the heart of the repository. It walks through the mindset, experiments, and math behind how the pipeline evolved from a baseline ruleset to its final state.
2. **Check `scripts/train.py` & `scripts/predict.py`**: Review the `RepairOrderTrainer` and `RepairOrderPredictor` classes to see how code is cleanly orchestrated.
3. **Review `src/repair_order/features.py`**: See how raw, unstructured German repair texts are tokenized, embedded, and transformed into numeric feature vectors.
4. **View outputs in `docs/text/`**: You can see exactly what the inference output looks like in the saved `predict_demo_out.txt` and `predict_batch_out.txt` logs.

## 📁 Repository Structure

```
.
├── scripts/                     # Core runnable scripts
│   ├── eda.py                   # Exploratory Data Analysis
│   ├── train.py                 # Core training pipeline (elevated Fully Mixed model)
│   └── predict.py               # Inference script
│
├── src/repair_order/            # Shared Python package
│   ├── config.py                # Constants (targets, keywords, makes)
│   ├── features.py              # Feature engineering functions
│   └── pipeline.py              # Pipeline load + predict utilities
│
├── docs/                        # Documentation & Reports
│   ├── project_evolution.md     # Detailed log of all experiments and optimizations
│   ├── markdowns/               # Component-level model documentation
│   └── assets/                  # Plots and images
│
├── experiments/                 # Experimental pipelines (G-BERT, etc.)
│   ├── combined_best/           # Incremental best-pipeline combinations
│   ├── gbert_base/              # German BERT embedding replacement experiment
│   ├── hailrepair_mae_exp/      # Skew-handling and MAE reduction strategies
│   └── log_transform/           # Log-transformation analysis
│
├── tests/                       # Pytest verification
├── .github/                     # CI/CD Workflows
├── Makefile                     # Task runner
├── pyproject.toml               # Python dependencies & config
└── CHANGELOG.md                 # Version history
```
