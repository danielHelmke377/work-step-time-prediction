# Work Step Time Prediction Pipeline

[![CI](https://github.com/danielHelmke377/work-step-time-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/danielHelmke377/work-step-time-prediction/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Two-stage machine learning pipeline for predicting **14 repair work steps** and their **execution time in hours** from unstructured JSON repair orders in the German automotive body-shop domain.

Built as a rapid assessment prototype and iterated toward production-oriented quality through multiple experiment cycles. See the [Project Evolution Summary](docs/project_evolution.md) for the full path from baseline to final pipeline.

---

## Why this project matters

Body-shop planning depends on estimating **which repair steps will happen** and **how long they will take** before work begins. This repository tackles that problem with a practical two-stage ML system:

- **Stage 1:** predict which work steps occur
- **Stage 2:** predict duration only for the work steps predicted active

That design keeps the problem aligned with the real business workflow and avoids predicting duration for steps that are unlikely to happen.

---

## What this project demonstrates

- **Multi-label → conditional regression cascade**  
  Stage 1 classifies work-step occurrence; Stage 2 predicts duration only for active steps.

- **NLP on German domain text**  
  Uses TF-IDF word + character n-grams, 17 hand-crafted regex keyword flags, and cost-center / price aggregation features.

- **Systematic ML experimentation**  
  Tracks the path from baseline rules through logistic regression, LightGBM, BERT-embedding variants, and combined best-pipeline experiments.

- **Engineering discipline**  
  Shared `src/repair_order/` package, editable install, tests, CI on Python 3.11 + 3.12, reproducible public workflow on synthetic data.

- **Public reproducibility despite private source data**  
  Includes a synthetic data generator so the full train → test → predict path runs end to end without NDA-restricted production data.

---

## At a glance

- **Problem:** predict 14 binary repair work steps and their durations from raw JSON repair orders
- **Domain:** German automotive body-shop / repair-order planning
- **Modeling strategy:** two-stage pipeline with LightGBM classification + per-target best regressor
- **Text features:** TF-IDF word n-grams, character n-grams, regex keyword flags
- **Structured features:** cost-center, time, price, and count aggregations
- **Public workflow:** synthetic-data generation, training, tests, and inference included
- **Validation:** strict single train/validation/test split on the available dataset

---

## Impact & results

The final prototype performs strongly for the target business context, with evaluation focused on frequency-weighted metrics so common work steps contribute proportionally to the final score.

| Metric | Value |
|---|---:|
| **Macro F1** | **0.838** |
| **Frequency-Weighted F1** | **0.935** |
| **Frequency-Weighted MAE** | **0.96 hrs** |
| **Frequency-Weighted Accuracy** | **0.943** |

> [!IMPORTANT]
> All reported metrics come from a **single strict train/validation/test split** with a 20% hold-out test set.  
> Cross-validation was **not** used in this prototype evaluation.

---

## Architecture

The core idea is a **two-stage pipeline** that separates *occurrence prediction* from *duration prediction*.

```mermaid
flowchart TD
    A[Raw Repair Order JSON] -->|Text & Positions| B[Feature Engineering Engine]
    B -->|TF-IDF N-grams, Regex Flags, Cost-Center Aggregations| C{Stage 1: Classifiers}

    C -->|LightGBM| D[14 Independent Binary Predictions]

    D -->|Target Active| E{Stage 2: Regressors}
    D -.-|Target Inactive| Z[Duration = 0.0 hrs]

    E -->|Ridge / LightGBM| F[Predicted Duration in Hours]

    F --> G([Final Repair Prediction])
    Z --> G
```

### Stage 1 — Multi-label occurrence prediction
Predicts binary presence (`0/1`) for each of the 14 work steps independently.

- Uses **`LGBMClassifier`** uniformly across all targets
- Optimizes **per-target decision thresholds** on the validation set using F1
- Combines text and structured features:
  - TF-IDF word n-grams
  - TF-IDF character n-grams
  - domain regex keyword flags
  - cost-center / price / time aggregations

### Stage 2 — Conditional duration prediction
Predicts duration **only** for targets predicted active by Stage 1.

- Uses a **per-target best regressor**
- Current pipeline selects from:
  - `Ridge`
  - `LGBMRegressor`
  - `ridge_auto`
- Targets with too few positive training samples fall back to the **mean of positive durations**

---

## Repository status

> [!NOTE]
> **Data privacy & reproducibility**  
> The proprietary customer dataset used for the original business results is **not included** in this public repository because of NDA and confidentiality constraints.
>
> This repository remains **fully reproducible on synthetic data**. You can generate synthetic orders, train the full two-stage pipeline, run tests, and execute predictions locally and in CI.

### Validated public workflow
The public workflow supports:

- synthetic data generation
- end-to-end training
- test execution
- batch prediction / explanation output
- CI validation on **Python 3.11 + 3.12**

---

## Quick start

> [!TIP]
> For the core pipeline, use **editable install from `pyproject.toml`**:
>
> ```bash
> pip install -e .[dev]
> ```
>
> `requirements.txt` is retained for compatibility with tools that expect a requirements file, but it is **not** the preferred setup path for the core workflow.

### 1) Create and activate a virtual environment

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
# Linux / macOS
pip install -e .[dev]

# Windows PowerShell
.\.venv\Scripts\python -m pip install -e .[dev]
```

### 3) Generate synthetic data

```bash
# Linux / macOS
python scripts/generate_synthetic_data.py

# Windows PowerShell
.\.venv\Scripts\python scripts/generate_synthetic_data.py
```

### 4) Train the pipeline

```bash
# Linux / macOS
python scripts/train.py --data data/synthetic_orders.json

# Windows PowerShell
.\.venv\Scripts\python scripts/train.py --data data/synthetic_orders.json
```

### 5) Run the test suite

```bash
# Linux / macOS
pytest -vv

# Windows PowerShell
.\.venv\Scripts\pytest -vv
```

### 6) Run batch inference

```bash
# Linux / macOS
python scripts/predict.py --batch 10

# Windows PowerShell
.\.venv\Scripts\python scripts/predict.py --batch 10
```

---

## How to review this repository in 5 minutes

If you are reviewing this project for an interview, code review, or portfolio assessment, this is the fastest path:

1. **Read this README first**  
   Understand the problem, results, architecture, and public workflow.

2. **Open the [Project Evolution Summary](docs/project_evolution.md)**  
   This is the best document for understanding *why* the final pipeline looks the way it does.

3. **Read the [Model Card](MODEL_CARD.md)**  
   See intended use, limitations, risk boundaries, and known failure modes.

4. **Inspect `scripts/train.py` and `scripts/predict.py`**  
   These show how the training and inference flow is orchestrated.

5. **Inspect `src/repair_order/features.py`**  
   This is where raw JSON repair-order content becomes model-ready features.

6. **Run the public synthetic workflow**  
   Generate synthetic data, train, test, and predict locally or via CI.

---

## Example prediction output

When running inference, the script produces an explainable report showing predicted work steps, confidence, and matched keywords.

```text
====================================================================
  WORK STEP TIME PREDICTION REPORT
====================================================================
  Make            : VOLKSWAGEN
  Line items      : 6
  Total input cost: EUR 1250.40

  TARGET                          ACTIVE    PROB  PRED(hrs)
  ------------------------------------------------------------
  Calibration (ADAS/cameras)         YES    0.94       1.50
  Body/chassis measurement           YES    0.68       2.00
  Dis-/mounting                      YES    0.99       4.20
  Body repair                        YES    0.87       3.40
  Painting — preparation             YES    0.83       1.20
  Painting — spraying                YES    0.82       2.30
  Glass replacement                  ---    0.11       0.00
  ...

  Total predicted repair time: 14.60 hrs

  EXPLANATION - Why each work step was predicted:
  ------------------------------------------------------------
  [Calibration (ADAS/cameras)]
    Keywords matched : kw_kalibrier, kw_sensor
  [Body/chassis measurement]
    Keywords matched : kw_karosserie, kw_vermessung
  [Painting — spraying]
    Keywords matched : kw_lack
====================================================================
```

---

## Engineering quality signals

This repository is intentionally structured as a **package-first, reproducible ML project** rather than a notebook dump.

- shared code in `src/repair_order/`
- editable install via `pyproject.toml`
- synthetic-data reproducibility
- test suite under `tests/`
- CI workflow that runs train/test/predict
- changelog and model card included
- experiment history documented in `docs/` and `experiments/`

---

## Limitations

- The original business metrics are based on **private, NDA-restricted source data**
- Evaluation uses a **single split**, not cross-validation
- The domain is specifically **German repair-order language and body-shop operations**
- Rare work steps remain harder to learn because positive examples are limited
- This is a **prototype-to-production-style portfolio project**, not a deployed SaaS system

For a fuller treatment of limitations, intended use, and failure modes, see the [Model Card](MODEL_CARD.md).

---

## Repository structure

```text
.
├── scripts/                        # Core runnable scripts
│   ├── generate_synthetic_data.py  # Public synthetic dataset generator
│   ├── train.py                    # Training pipeline
│   ├── predict.py                  # Inference script with explanations
│   └── eda.py                      # Exploratory data analysis
│
├── src/repair_order/               # Shared Python package
│   ├── config.py                   # Constants (targets, keywords, makes)
│   ├── features.py                 # Feature engineering functions
│   └── pipeline.py                 # Pipeline load + predict utilities
│
├── docs/                           # Documentation & reports
│   ├── project_evolution.md        # Experiment history and optimization journey
│   ├── markdowns/                  # Component-level documentation
│   └── assets/                     # Images / plots
│
├── experiments/                    # Experimental pipelines
│   ├── combined_best/
│   ├── gbert_base/
│   ├── hailrepair_mae_exp/
│   └── log_transform/
│
├── tests/                          # Pytest verification
├── .github/                        # CI workflows
├── Makefile                        # Task runner
├── pyproject.toml                  # Dependencies and tool config
├── requirements.txt                # Compatibility requirements file
├── MODEL_CARD.md                   # Model documentation and limitations
└── CHANGELOG.md                    # Version history
```

---

## Related documentation

- [Project Evolution Summary](docs/project_evolution.md)
- [Model Card](MODEL_CARD.md)
- [`tests/README.md`](tests/README.md)
- [`CHANGELOG.md`](CHANGELOG.md)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
