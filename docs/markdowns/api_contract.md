# API Contract — Work Step Time Prediction Service

> **Status: Design only. No code written yet.**
> This document is the specification that the implementation phase must follow exactly.

---

## 0. Time units

All time values in both requests and responses are in **hours**, consistent with
the pipeline's internal key names (`predicted_hours`, `total_input_time_hrs`,
`total_input_time_hrs`). The `(min)` column headers in `predict_batch_out.txt`
are a known labelling mistake in that output file and do not reflect the actual unit.

---

## 1. Endpoint Summary

| # | Method | Path | Purpose |
|---|---|---|---|
| 1 | `GET` | `/health` | Liveness + readiness check (is the model loaded?) |
| 2 | `POST` | `/predict` | Single-order inference — the primary endpoint |
| 3 | `GET` | `/model-info` | Static metadata about the loaded model artifact |

Base path: none (all paths are at root). Version prefix (`/v1/`) is optional and
can be added later without breaking the internal design.

---

## 2. Decision: Single Order vs Batch

**Recommendation: support exactly one repair order per request.**

Reasons:
- `predict_order()` is a single-record function. Batching is trivially a loop
  in the client — the library does not vectorise across records.
- A single-order contract is far easier to document, test, and demonstrate.
- Portfolio context: one clear request / one clear response is the most legible
  design for a recruiter reviewing the Swagger UI.
- If batch support is desired later it can be added as `POST /predict/batch`
  without changing the single-order contract.

---

## 3. POST /predict

### 3.1 Purpose

Accept a single raw repair order, run the two-stage pipeline, and return
structured predictions for all 14 work steps with optional keyword explanation.

### 3.2 Request Schema

```json
POST /predict
Content-Type: application/json

{
  "make": "VOLKSWAGEN",
  "calculated_positions": [
    {
      "text": "HAGELSCHADENREPARATUR: METHODE ...",
      "total_time": 120.0,
      "total_price": 250.0,
      "cost_center": "bodywork"
    }
  ],
  "explain": false
}
```

#### Field-by-field specification

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `make` | `string` | No | `"unknown"` | Vehicle manufacturer name. Case insensitive; normalised to UPPER internally. Null or missing → `"unknown"`. |
| `calculated_positions` | `array[Position]` | **Yes** | — | Must contain at least 1 element. Empty array → 422. |
| `explain` | `boolean` | No | `false` | If `true`, include keyword explanation in the response. |

#### Position object

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `text` | `string` | No | `""` | Position description text. Null → empty string. Leading/trailing whitespace stripped. |
| `total_time` | `number` | No | `0.0` | Duration of this position. Null or missing → `0.0`. Negative values → clamped to `0.0`. |
| `total_price` | `number` | No | `0.0` | Price of this position. Null or missing → `0.0`. Negative values → clamped to `0.0`. |
| `cost_center` | `string` | No | `"unknown_cc"` | Generic cost-center label (e.g. `"bodywork"`, `"painting"`, `"hail"`). Null or unrecognised value → `"unknown_cc"` (no error). |

#### Field name mapping to internal pipeline

The API uses `snake_case` names (`calculated_positions`, `total_time`,
`total_price`, `cost_center`). Internally, `predict_order()` expects the raw
domain dict format (`calculatedPositions`, `totalTime`, `totalPrice`,
`genericCostCenter`). The adapter in `app/predictor.py` performs this mapping
transparently — neither `predict_order()` nor `features.py` are modified.

```
API field             → internal dict key
─────────────────────────────────────────
make                  → record["input"]["make"]
calculated_positions  → record["input"]["calculatedPositions"]
  .text               →   position["text"]
  .total_time         →   position["totalTime"]
  .total_price        →   position["totalPrice"]
  .cost_center        →   position["genericCostCenter"]
```

#### Extra / unknown fields

Unknown fields in the top-level request object and inside `Position` objects are
**silently ignored** (Pydantic `model_config = {"extra": "ignore"}`). This keeps
the API forward-compatible if the caller sends additional metadata.

### 3.3 Successful Response Schema (HTTP 200)

```json
{
  "make": "VOLKSWAGEN",
  "n_positions": 4,
  "input_summary": {
    "total_time_hrs": 10.89,
    "total_price": 3230.59
  },
  "predictions": {
    "calibration": {
      "label": "Calibration (ADAS/cameras)",
      "active": false,
      "probability": 0.0,
      "threshold": 0.42,
      "predicted_time": 0.0
    },
    "hailrepair": {
      "label": "Hail repair",
      "active": true,
      "probability": 1.0,
      "threshold": 0.35,
      "predicted_hours": 10.81
    }
    // ... all 14 targets always present
  },
  "active_steps": ["hailrepair", "assembly", "cleaning", "paintingFinish", "glas"],
  "total_predicted_hours": 10.89,
  "elapsed_ms": 52.3,
  "explanation": null
}
```

When `explain: true` is sent, `explanation` is populated:

```json
"explanation": {
  "hailrepair": {
    "triggered_keywords": ["hagel", "dellen"],
    "matching_positions": [
      "HAGELSCHADENREPARATUR: METHODE ...",
      "DACH (AUSSENHAUT) (GROESSE: 20 MM) (80 DELLEN)"
    ]
  },
  "assembly": {
    "triggered_keywords": ["montage"],
    "matching_positions": ["MOTORHAUBE A+E"]
  }
}
```

#### Success response field specification

| Field | Type | Notes |
|---|---|---|
| `make` | `string` | Normalised make as used in prediction |
| `n_positions` | `integer` | Count of positions after cleaning (empty rows removed) |
| `input_summary.total_time_hrs` | `float` | Sum of `total_time` across all positions, in hours |
| `input_summary.total_price` | `float` | Sum of `total_price` across all positions |
| `predictions` | `object` | Keys are internal target IDs, always all 14 present |
| `predictions[t].label` | `string` | Human-readable label from `config.TARGET_LABELS` |
| `predictions[t].active` | `boolean` | Whether Stage 1 threshold was exceeded |
| `predictions[t].probability` | `float [0,1]` | Raw classifier probability (4 decimal places) |
| `predictions[t].threshold` | `float [0,1]` | Per-target optimised F1 threshold (2 decimal places) |
| `predictions[t].predicted_hours` | `float ≥ 0` | Predicted duration in hours; `0.0` when `active=false` |
| `active_steps` | `string[]` | Ordered list of target IDs where `active=true` |
| `total_predicted_hours` | `float ≥ 0` | Sum of `predicted_hours` across all targets, in hours |
| `elapsed_ms` | `float` | Wall-clock inference time in milliseconds |
| `explanation` | `object \| null` | Present and populated only when `explain=true`; otherwise `null` |

**Why expose `probability` and `threshold` separately?**
A caller can use the raw probability to build their own risk display or adjust
threshold sensitivity. Exposing only a binary `active` flag is insufficient for
a production-grade API. This also clearly shows the f1-tuned thresholds are per-target.

**Why label in the per-target object instead of only in a separate lookup?**
The response must be self-contained. A caller should not need to join on a
separate lookup table to render the result.

---

## 4. GET /health

### 4.1 Purpose

Used by load balancers, orchestrators (Kubernetes, Docker Compose healthcheck),
and monitoring to verify that the process is alive and the model is ready.

**Decision:** report both *liveness* and *readiness* in a single endpoint.
For a portfolio project one endpoint is cleaner; distinguishing `/health/live`
and `/health/ready` is an enterprise pattern that adds complexity without
benefit here.

### 4.2 Response Schema

#### Model loaded and ready (HTTP 200)
```json
{
  "status": "ok",
  "model_loaded": true,
  "n_targets": 14
}
```

#### Process alive but model not loaded (HTTP 503)
```json
{
  "status": "degraded",
  "model_loaded": false,
  "n_targets": 0
}
```

**Why 503 when degraded?** A 200 with `model_loaded: false` would pass load
balancer health checks and route real traffic to a broken instance. 503 forces
orchestrators to restart or exclude the pod.

| Field | Type | Notes |
|---|---|---|
| `status` | `string` | `"ok"` or `"degraded"` |
| `model_loaded` | `boolean` | Whether the pipeline pickle was loaded successfully at startup |
| `n_targets` | `integer` | Number of output targets in the loaded pipeline; `0` if not loaded |

---

## 5. GET /model-info

### 5.1 Purpose

Returns stable metadata about the loaded model artifact. Useful for
audit trails, version checks, and making the Swagger UI self-documenting.

### 5.2 Response Schema (HTTP 200)

```json
{
  "model_version": "2.0.0",
  "trained_on": "2026-03-18",
  "pipeline_file": "models/two_stage_pipeline.pkl",
  "time_unit": "hours",
  "n_targets": 14,
  "targets": [
    {"id": "calibration",      "label": "Calibration (ADAS/cameras)"},
    {"id": "wheelmeasurement", "label": "Wheel alignment measurement"},
    {"id": "bodymeasurement",  "label": "Body/chassis measurement"},
    {"id": "dismounting",      "label": "Dis-/mounting"},
    {"id": "bodyrepair",       "label": "Body repair"},
    {"id": "assembly",         "label": "Assembly"},
    {"id": "plasticrepair",    "label": "Plastic repair"},
    {"id": "allTiresService",  "label": "Tyre service"},
    {"id": "cleaning",         "label": "Cleaning"},
    {"id": "paintingPreparation", "label": "Painting — preparation"},
    {"id": "paintingSpraying", "label": "Painting — spraying"},
    {"id": "paintingFinish",   "label": "Painting — finish"},
    {"id": "hailrepair",       "label": "Hail repair"},
    {"id": "glas",             "label": "Glass replacement"}
  ],
  "feature_groups": [
    "tfidf_word_ngrams",
    "tfidf_char_ngrams",
    "numeric_aggregates",
    "keyword_flags",
    "make_encoding"
  ],
  "stage1_classifier": "LGBMClassifier",
  "stage2_regressors": ["LGBMRegressor", "Ridge", "mean_fallback"]
}
```

#### Source of each field

| Field | Source |
|---|---|
| `model_version` | `models/model_info.json` (written by `train.py`) |
| `trained_on` | `models/model_info.json` |
| `pipeline_file` | Relative path of the loaded `.pkl`, computed at startup |
| `n_targets` | `len(pipeline["output_targets"])` at startup |
| `targets[].id` | `pipeline["output_targets"]` |
| `targets[].label` | `config.TARGET_LABELS[id]` |
| `feature_groups` | Static — reflects the known feature engineering path |
| `stage1_classifier` | Static — reflects current training config |
| `stage2_regressors` | Static — reflects current training config |

**Why not expose thresholds or model hyperparameters here?**
Those are internal implementation details. They appear per-target in the `predict`
response where they are actionable. The `/model-info` endpoint is for version and
capability metadata only.

---

## 6. Error Response Schemas

All error responses share the same envelope:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Human-readable description of the error",
    "details": [...]
  }
}
```

`details` is only present when there are multiple sub-errors (e.g. Pydantic
field validation failures). It is omitted or `null` for single-error responses.

### 6.1 Request Validation Failure (HTTP 422)

Triggered by: missing required fields, wrong types, empty `calculated_positions`.

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request body failed validation.",
    "details": [
      {
        "field": "calculated_positions",
        "issue": "List must contain at least 1 item."
      },
      {
        "field": "calculated_positions[2].total_price",
        "issue": "Value must be a number."
      }
    ]
  }
}
```

### 6.2 Model Not Ready (HTTP 503)

Triggered by: `POST /predict` called before the model loaded (startup failure or
race condition).

```json
{
  "error": {
    "code": "MODEL_NOT_READY",
    "message": "The prediction model is not loaded. Check /health for status.",
    "details": null
  }
}
```

### 6.3 Internal Server Error (HTTP 500)

Triggered by: unexpected exception inside `predict_order()` or the adapter.

```json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "An unexpected error occurred during inference. Please check the server logs.",
    "details": null
  }
}
```

> [!CAUTION]
> Internal stack traces must **never** appear in the HTTP response body.
> They should be logged server-side only. `INTERNAL_ERROR` responses must not
> leak any model internals, pickle paths, or filesystem details.

---

## 7. Validation Rules (Edge Cases)

| Scenario | Behaviour |
|---|---|
| `calculated_positions` is an empty array `[]` | **422** — "List must contain at least 1 item." |
| `calculated_positions` is missing from the body | **422** — required field |
| All positions have `text: ""` (blank strings) | **Allowed.** `preprocess_positions()` filters blanks internally; prediction runs on numeric features only. Not a 422. |
| `text` is `null` | Treated as `""` (coerced by Pydantic default). |
| `total_time` is `null` | Treated as `0.0` (coerced by Pydantic default). |
| `total_price` is `null` | Treated as `0.0` (coerced by Pydantic default). |
| `total_price` or `total_time` is negative | Clamped to `0.0` in the adapter (mirrors `preprocess_positions` behaviour). Not a 422. |
| `total_price` is `"abc"` (non-numeric string) | **422** — type validation failure. |
| `cost_center` is unrecognised (e.g. `"unknown_thing"`) | Accepted silently. Passed through as-is to `preprocess_positions`, treated as `unknown_cc` where needed. |
| `make` is `null` | Treated as `"unknown"`. |
| `make` is an empty string `""` | Treated as `"unknown"`. |
| Extra top-level keys (e.g. `"order_id": "ABC"`) | Silently ignored (`extra = "ignore"`). |
| Extra keys inside a `Position` object | Silently ignored. |
| `explain` missing | Defaults to `false`. |
| `explain` is a non-boolean (e.g. `"yes"`) | **422** — type validation failure. |
| All positions filtered out after preprocessing | Model runs on a degenerate feature vector (all zeros). Returns a valid prediction with all targets `active: false` and `total_predicted_time: 0.0`. Not a 422. This mirrors current CLI behaviour. |

---

## 8. Recommended HTTP Status Codes

| Scenario | Code |
|---|---|
| Successful prediction | `200 OK` |
| Successful health check (model ready) | `200 OK` |
| Health check (model not ready) | `503 Service Unavailable` |
| Successful model-info retrieval | `200 OK` |
| Request validation failure | `422 Unprocessable Entity` |
| Model not loaded when predict called | `503 Service Unavailable` |
| Unexpected internal error | `500 Internal Server Error` |
| Method not allowed (e.g. `GET /predict`) | `405 Method Not Allowed` (FastAPI handles automatically) |
| Path not found | `404 Not Found` (FastAPI handles automatically) |

FastAPI's default OpenAPI error format returns a `detail` key. The API overrides
the default exception handlers to use the `error` envelope above for all error
paths, so every error response is consistent.

---

## 9. Open Design Decisions

### 9.1 `model_info.json` sidecar format

The sidecar file `models/model_info.json` does not exist yet. It will be
created by modifying `scripts/train.py` to write it after saving the pickle.
The `/model-info` endpoint must gracefully handle the case where this file is
missing (i.e. pipeline was trained with an older version of `train.py`) and
return what it can from the in-memory pipeline dict.

### 9.2 Response field name style for target IDs

The `predictions` object uses **internal target ID strings** (e.g.
`"hailrepair"`, `"calibration"`) as keys, matching `config.OUTPUT_TARGETS`.
These are the same strings the CLI uses internally. The human-readable label
is a sub-field (`label`). This keeps the response machine-parseable by ID
while being human-readable in the same structure.

---

## 10. Minimal Example Exchange (Annotated)

### Request

```http
POST /predict HTTP/1.1
Content-Type: application/json

{
  "make": "MAZDA",
  "calculated_positions": [
    {
      "text": "HAGELSCHADENREPARATUR: METHODE HAGELSCHADENZENTRUM / BVAT",
      "total_time": 638.10,
      "total_price": 2800.00,
      "cost_center": "hail"
    },
    {
      "text": "FAHRZEUGREINIGUNG",
      "total_time": 1.58,
      "total_price": 40.00,
      "cost_center": "others"
    }
  ],
  "explain": true
}
```

### Response (HTTP 200)

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "make": "MAZDA",
  "n_positions": 2,
  "input_summary": {
    "total_time_hrs": 10.66,
    "total_price": 2840.00
  },
  "predictions": {
    "calibration":         {"label": "Calibration (ADAS/cameras)", "active": false, "probability": 0.0,  "threshold": 0.42, "predicted_hours": 0.0},
    "wheelmeasurement":    {"label": "Wheel alignment measurement", "active": false, "probability": 0.0,  "threshold": 0.38, "predicted_hours": 0.0},
    "bodymeasurement":     {"label": "Body/chassis measurement",   "active": false, "probability": 0.0,  "threshold": 0.40, "predicted_hours": 0.0},
    "dismounting":         {"label": "Dis-/mounting",              "active": false, "probability": 0.12, "threshold": 0.15, "predicted_hours": 0.0},
    "bodyrepair":          {"label": "Body repair",                "active": false, "probability": 0.0,  "threshold": 0.45, "predicted_hours": 0.0},
    "assembly":            {"label": "Assembly",                   "active": false, "probability": 0.08, "threshold": 0.20, "predicted_hours": 0.0},
    "plasticrepair":       {"label": "Plastic repair",             "active": false, "probability": 0.0,  "threshold": 0.50, "predicted_hours": 0.0},
    "allTiresService":     {"label": "Tyre service",               "active": false, "probability": 0.0,  "threshold": 0.48, "predicted_hours": 0.0},
    "cleaning":            {"label": "Cleaning",                   "active": true,  "probability": 1.0,  "threshold": 0.30, "predicted_hours": 1.58},
    "paintingPreparation": {"label": "Painting — preparation",     "active": false, "probability": 0.0,  "threshold": 0.35, "predicted_hours": 0.0},
    "paintingSpraying":    {"label": "Painting — spraying",        "active": false, "probability": 0.0,  "threshold": 0.38, "predicted_hours": 0.0},
    "paintingFinish":      {"label": "Painting — finish",          "active": true,  "probability": 1.0,  "threshold": 0.25, "predicted_hours": 0.20},
    "hailrepair":          {"label": "Hail repair",                "active": true,  "probability": 1.0,  "threshold": 0.35, "predicted_hours": 10.81},
    "glas":                {"label": "Glass replacement",          "active": false, "probability": 0.05, "threshold": 0.42, "predicted_hours": 0.0}
  },
  "active_steps": ["cleaning", "paintingFinish", "hailrepair"],
  "total_predicted_hours": 12.59,
  "elapsed_ms": 48.2,
  "explanation": {
    "cleaning": {
      "triggered_keywords": ["reinigung"],
      "matching_positions": ["FAHRZEUGREINIGUNG"]
    },
    "paintingFinish": {
      "triggered_keywords": ["klebetechnik"],
      "matching_positions": []
    },
    "hailrepair": {
      "triggered_keywords": ["hagel"],
      "matching_positions": ["HAGELSCHADENREPARATUR: METHODE HAGELSCHADENZENTRUM / BVAT"]
    }
  }
}
```
