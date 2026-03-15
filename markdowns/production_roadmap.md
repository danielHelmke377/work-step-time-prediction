# Production Readiness Roadmap

**Script:** `code/predict.py` → production API  
**Date:** 2026-03-15

---

## Context & Scope Note

> [!IMPORTANT]
> This project was built as a **time-boxed technical prototype** (4-hour assessment window). The following reflects how the pipeline would differ if developed for actual production deployment.

In a real-world engagement, the following would be substantially expanded **before** any model is selected or trained:

- **Feature engineering** receives significantly more iteration — domain expert interviews, systematic ablation studies, and validation that each feature group adds measurable lift on held-out data.
- **Model selection** involves a structured comparison across more algorithm families, with cross-validated hyperparameter search rather than a single train/val/test pass.
- **Data quality checks** are implemented at the **ingestion stage** — before any feature computation — to catch upstream schema drift, missing mandatory fields, outlier proportions outside historical ranges, and duplicate records. In production, a record that fails quality gates is rejected (or quarantined) before it ever reaches the model.
- **Every procedure** (preprocessing, feature extraction, threshold tuning, serialisation) would be covered by a comprehensive automated test suite — unit tests, integration tests, and regression tests — not just the lightweight smoke tests present here.
- **New pipelines are never cut over immediately.** A standard **4-week parallel-run phase** is used: the new model runs alongside the existing system, its predictions are logged but not acted upon, and its outputs are compared against production ground truth before any live traffic is switched over. Only after this validation period is the new pipeline promoted.

---

## What Needs to Change for Production

### 🔴 Must-have — Blockers

#### 1. Serving Layer — Wrap the Model in an API

`predict.py` is a CLI script. Production needs an HTTP endpoint:

```python
# FastAPI example
from fastapi import FastAPI
from repair_order.pipeline import load_pipeline, predict_single

app = FastAPI()
pipeline = load_pipeline()

@app.post("/predict")
def predict(order: dict):
    return predict_single(pipeline, order)
```

- Containerise with **Docker** so it runs identically in every environment
- Expose a `/health` endpoint so orchestrators (Kubernetes, ECS) can confirm the model is loaded

#### 2. Model Registry / Versioned Artifact Storage

The `.pkl` is currently a local file excluded from git. In production:

- Store trained artifacts in **S3 / Azure Blob / GCS** with versioned paths:  
  `models/v1.3.0/two_stage_pipeline.pkl`
- The API loads from the registry at startup — not from a local path
- Enables instant rollback: switch back to the previous version by changing one config value

#### 3. Input Validation & Schema Contract

The API must reject malformed inputs cleanly before they reach the model:

```python
from pydantic import BaseModel

class RepairOrder(BaseModel):
    input: dict            # validates required keys exist
    output: dict | None = None  # optional for inference
```

Without this, bad inputs produce cryptic 500 errors instead of clear 422 validation responses.

#### 4. Comprehensive Test Suite

Current smoke tests only verify the pickle structure. Production requires:

- **Unit tests** for `preprocess_positions`, `build_numeric_features`, `build_order_text`
- **Integration test:** send a known order → assert specific predictions match a golden reference
- **Data-drift tests:** assert input feature distributions haven't shifted beyond a defined threshold since the last training run

---

### 🟡 High-value — Should-have

#### 5. Monitoring & Alerting

Log every prediction (input fingerprint, predicted targets, confidence scores, latency) to a time-series store. Alert when:

- Macro F1 on a rolling window drops below 0.80
- p95 prediction latency exceeds 200 ms
- Any input field is null more than X% of requests (upstream data quality signal)

#### 6. Automated Retraining Pipeline

Schedule a periodic job (weekly or monthly) that:
1. Pulls fresh labelled orders
2. Retrains the full pipeline with fixed hyperparameters
3. Evaluates on a held-out test set
4. If new model beats the current production model by > 2 pp Macro F1 → promote to staging
5. Run the 4-week parallel-run phase before cutting over to live traffic

Tools: **GitHub Actions** (simple), **Apache Airflow** or **Prefect** (complex).

#### 7. Configuration & Secrets Management

Environment-specific values (S3 bucket, DB connection string, model version to load) should come from environment variables or a secrets manager (AWS Secrets Manager, HashiCorp Vault) — not from hardcoded constants.

#### 8. Canary / Shadow Deployment

When deploying a new model version, first route 5% of live traffic to it and compare its predictions to the current model's — without the new model's output affecting any downstream system. Promote only if metrics match or improve.

---

### 🟢 Nice-to-have — Professional Polish

#### 9. Data Versioning (DVC)

Track which exact dataset version trained which model version. Ensures any historical model can be reproduced exactly from its training data snapshot.

#### 10. Explainability Endpoint (`/explain`)

Expose `explain_prediction()` as a separate endpoint for human reviewers. In the body-shop context, estimators need to understand *why* a work step was predicted — especially for high-cost positions like `hailrepair`.

---

## Summary

| Layer | Current State | Production Requirement |
|---|---|---|
| Serving | CLI script | FastAPI + Docker container |
| Artifact storage | Local `.pkl` (git-ignored) | S3/Blob with version tagging |
| Input validation | None | Pydantic schema + rejection logic |
| Testing | Smoke tests only | Unit + integration + drift tests |
| Monitoring | None | Latency, F1, data quality alerts |
| Data quality | None | Ingestion-stage validation gates |
| Retraining | Manual | Scheduled pipeline with auto-promotion |
| Config | Hardcoded constants | Environment variables / secrets manager |
| Deployment | `git push` | CI/CD with 4-week parallel-run phase |
| Explainability | CLI only | `/explain` API endpoint |

The highest-impact single step is **Serving + Model Registry (#1 + #2)**: wrap `predict_single()` in a FastAPI app, containerise it, and store the model in cloud storage. That moves the project from "script on a laptop" to "something deployable and versioned."
