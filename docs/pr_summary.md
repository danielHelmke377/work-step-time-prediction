# PR: Portfolio-Ready Engineering Pass — work-step-time-prediction

## Summary

This PR brings the repository from a working prototype to a maintainability- and reviewer-ready state. It hardens the public synthetic-data workflow, aligns all documentation with the actual implementation, adds a real unit-test layer, enforces a lint gate in CI, and removes the visible presentation debt that would cause an external reviewer to doubt the project's engineering discipline.

No core model behavior was changed. All metrics on the original business data remain as reported.

---

## Changes by Area

### Setup / Packaging

| File | Change |
|---|---|
| `pyproject.toml` | Version `1.3.0` → `2.0.0` (matches CHANGELOG); `ruff>=0.4` added to `[dev]`; `[tool.ruff]` + `[tool.ruff.lint]` config added; `[tool.pytest.ini_options]` added (testpaths, addopts) |
| `requirements.txt` | Removed `gbert` extras (`torch`, `transformers`, `sentence-transformers`); added explanatory comments |

### Docs / Consistency

| File | Change |
|---|---|
| `README.md` | Replaced hedging intro with confident factual pitch; added "What This Demonstrates" section (5 bullets for hiring managers/reviewers); fixed repo structure map; removed `make predict` reference |
| `scripts/train.py` | Docstring: removed non-existent `--use-logreg` flag reference; removed dead experiment-file references; documented real `BEST_REG`/`ridge_auto` logic; swapped default data priority to `synthetic_orders.json` |
| `MODEL_CARD.md` | Added actual metric results table (Macro F1 0.838, Weighted F1 0.935, MAE 0.96 hrs, Acc 0.943); NDA context note; restructured into Results + Methodology subsections |
| `tests/README.md` | Full rewrite: added `test_functional.py` section; updated run command to `pytest -vv`; documented gbert skip policy; added coverage summary table |
| `tests/test_smoke.py` | Fixed outdated docstring run command (`pytest tests/ -v` → `pytest -vv`) |
| `CHANGELOG.md` | Fixed misleading "replacing `requirements.txt`" → "complementing" |
| `src/repair_order/features.py` | Removed garbled duplicate sentence from docstring |
| `conftest.py` | Updated docstring to reflect canonical `pip install -e .[dev]` as install path |
| `src/repair_order/config.py` | Removed garbled duplicate comment |
| `src/repair_order/pipeline.py` | Removed unused `import numpy as np` |

### Tests

| File | Change |
|---|---|
| `tests/test_unit.py` *(new)* | 10 deterministic unit tests — no model loading required: `TestPreprocessPositions` (3 tests), `TestBuildOrderText` (3 tests), `TestBuildNumericFeatures` (4 tests) |
| `generate_synthetic_data.py` | Increased orders to 500; forced 10 samples/target for stratification stability |

**Test count: 8 → 18 passed, 4 skipped (gbert — expected and documented)**

### CI

| File | Change |
|---|---|
| `.github/workflows/ci.yml` | Added `Lint (ruff)` step after install; expanded Python matrix to `["3.11", "3.12"]`; added full synthetic-data workflow steps (generate → train → predict) |

### Developer Tooling

| Tool | Status |
|---|---|
| `ruff` | Added to `[dev]` extras; configured in `pyproject.toml`; enforced in CI |
| `pytest` | `[tool.pytest.ini_options]` configured — `pytest` with no args works from repo root |

### Repo Polish

| Item | Change |
|---|---|
| `Makefile` | Rewritten to be POSIX-compliant with active-venv assumption; added `make data`, `make all` |
| `refactor_predict.py` | Deleted (stale unreferenced root-level file) |
| `scripts/inspect_pipeline.py` | Removed (untracked debug helper, never canonical) |
| `scripts/predict.py` | Import block sorted by ruff |

---

## Why This Matters

| Dimension | Before | After |
|---|---|---|
| **Reviewer trust** | Docstrings referenced non-existent flags; MODEL_CARD had no numbers | All docs match the code; MODEL_CARD has the actual metric results |
| **Reproducibility** | CI only ran smoke tests | CI runs full generate → train → predict on every push, on Python 3.11 + 3.12 |
| **Maintainability** | Zero unit tests; no linter | 10 unit tests document the feature contract; ruff enforced in CI |
| **Engineering confidence** | `pyproject.toml` version 3 major versions behind; no quality tooling in dev extras | Version matches CHANGELOG; `pip install -e .[dev]` gives both pytest and ruff |

---

## Validation Performed

```
ruff check src/ scripts/ tests/        → All checks passed
pytest -q
  22 collected: 18 passed, 4 skipped  → Exit code: 0
  (4 skipped: TestGbertExperiment — gbert artefacts not generated, expected)
```

---

## Deferred Follow-ups (Intentionally Not Included)

| Item | Rationale for deferral |
|---|---|
| Windows CI job | Medium risk: path separators, encoding, no make. README notes are sufficient |
| Python 3.10 in CI matrix | Low traffic; low priority |
| Type annotations (`features.py`, `pipeline.py`) | Medium effort, low reviewer-facing value |
| `CONTRIBUTING.md` | Optional for a portfolio project |
| `requirements-lock.txt` | Low priority; editable install is canonical |
| `ruff --fix` as pre-commit hook | Nice to have; low priority |
