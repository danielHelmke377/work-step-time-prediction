# tests/ — Smoke Tests

Run with:

```bash
pytest tests/ -v
```

Tests check that saved pipeline artefacts exist, load correctly, and have the expected
structure. They do **not** require the full dataset — only the saved pickles.

| Test class | What it checks |
|---|---|
| `TestCorePipeline` | 14 targets, classifiers, regressors, thresholds in range, TF-IDF vectorisers |
| `TestGbertExperiment` | BERT embeddings shape (N×768), L2-normalisation, gbert pipeline metadata |

> Tests skip automatically if the relevant artefact has not yet been generated.
> Run `model_phase2.py` and/or `gbert_base/code/model_gbert.py` first.
