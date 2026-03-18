# tests/ — Test Suite

Run with:

```bash
# (with active venv)
pytest -vv
```

Tests are split into two files with complementary coverage:

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

## Coverage Summary

| Area | Covered by |
|---|---|
| Package imports | CI import check (`ci.yml`) |
| Artefact structure / integrity | `test_smoke.py` |
| Full synthetic train + predict | `test_functional.py` |
| gbert experiment artefacts | `test_smoke.py::TestGbertExperiment` (skipped if absent) |
| Core model logic | *(not unit-tested — metrics are validated in `docs/markdowns/training_results.md`)* |
