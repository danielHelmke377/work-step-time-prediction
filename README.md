# Repair Order Prediction Pipeline

[![CI](https://github.com/danielHelmke377/work-step-time-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/danielHelmke377/work-step-time-prediction/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predicts **14 binary work steps** (e.g., `bodyrepair`, `paintingSpraying`) and their **duration in hours** from unstructured JSON repair order data. 

This repository represents the **first step in prototyping the best possible model** for a multi-label and regression pipeline, evolving rapidly from a 4-hour assessment into a highly optimized architecture. While not yet a fully integrated production system, it demonstrates the systematic iteration required to build one.

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

## 💻 Quick Start

### 1. Requirements

Ensure you have Python 3.9+ installed.

```bash
git clone https://github.com/danielHelmke377/work-step-time-prediction.git
cd work-step-time-prediction

make setup
```

### 2. Add Data

Place the `orders_simplified_sample.json` data file in `data/`.

### 3. Execution

Use the provided `Makefile` to run the core operations:

```bash
# Run the core training pipeline (Stage 1 & Stage 2)
make train

# Run inference batch on 10 random demo orders
make predict

# Run unit tests
make test
```

*Alternatively, activate your environment and run scripts directly:*
```bash
python scripts/train.py
python scripts/predict.py --demo 42
```

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
