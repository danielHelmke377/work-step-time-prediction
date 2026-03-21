# tests/ — Test Suite

Run with:

```bash
# (with active venv)
pytest -vv
```

Tests are split across four files:

---

## `test_smoke.py` — Integrity / Artefact Tests

Check that saved artefacts load correctly and have the expected structure.
**No training required** — only the saved pipeline pickle.

| Test class | What it checks |
|---|---|
| `TestCorePipeline` | Pipeline loads; `output_targets` non-empty; classifiers, regressors, and thresholds present for every trained target; TF-IDF vectorisers present |
| `TestGbertExperiment` | BERT embeddings shape (N×768), L2-normalisation, gbert pipeline metadata |

> [!NOTE]
> `TestGbertExperiment` tests skip automatically if the gbert artefacts have not been generated.
> Run `gbert_base/code/extract_bert_embeddings.py` and `gbert_base/code/model_gbert.py` to enable them.

---

## `test_functional.py` — End-to-End Functional Tests

Verify the **full public synthetic-data workflow** end to end.
These tests run `scripts/train.py` as a subprocess against `data/synthetic_orders.json`
and then exercise the resulting pipeline through the `repair_order` package API.

| Test | What it checks |
|---|---|
| `test_pipeline_has_all_targets` | Pipeline trained on synthetic data contains all 14 declared output targets |
| `test_prediction_returns_nonempty_dict` | `predict_order()` returns a valid prediction dict with the correct keys and a non-negative total predicted time |

> [!NOTE]
> `test_functional.py` automatically generates `data/synthetic_orders.json` if it does not exist.
> These tests are the canonical public reproducibility gate for this repository.

---

## `test_api.py` — API Integration Tests

Validate the FastAPI inference service end-to-end.
Tests start the API with the trained champion model and exercise all three endpoints.

| Test class | What it checks |
|---|---|
| `TestHealth` | `/health` returns 200, correct status/model_loaded/n_targets fields |
| `TestModelLoaded` | `/predict` returns correct shape, all 14 targets, required fields, non-negative hours, elapsed_ms |
| `TestPredictEdgeCases` | Empty positions, single position, explain flag, missing optional fields |
| `TestModelInfo` | `/model-info` returns version, targets list, feature groups |
| `TestErrorHandling` | Validation errors → 422, unknown cost_center handled |

---

## `test_lifecycle.py` — Champion-Challenger Lifecycle Tests

Validate the model promotion workflow without training.
All tests run in milliseconds using synthetic metric fixtures.

| Test | What it checks |
|---|---|
| Error handling | Missing champion/challenger, malformed JSON, missing required keys |
| Rule evaluation | R1/R2/R3 pass and fail boundaries with known metrics |
| Dry-run guarantee | No files written in dry-run mode |
| Promotion flow | Decision artifacts created, archive slot populated |
| Rejection flow | Champion artifacts unchanged when rules fail |
| Report content | Generated markdown includes all expected sections |

---

## Coverage Summary

| Area | Covered by |
|---|---|
| Package imports | CI import check (`ci.yml`) |
| Artefact structure / integrity | `test_smoke.py` |
| Full synthetic train + predict | `test_functional.py` |
| API endpoints (health / predict / model-info) | `test_api.py` (23 tests) |
| Champion-challenger lifecycle | `test_lifecycle.py` (14 tests) |
| gbert experiment artefacts | `test_smoke.py::TestGbertExperiment` (skipped if absent) |
| Core model logic | *(not unit-tested — metrics are validated in `docs/markdowns/training_results.md`)* |
