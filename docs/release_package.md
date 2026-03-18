# Release / Handoff Package — work-step-time-prediction

---

## Maintainer / Reviewer Checklist

### Documentation
- [x] `README.md` — confident pitch, "What This Demonstrates", complete Quick Start, architecture diagram, structure map
- [x] `MODEL_CARD.md` — actual metric results, NDA context, intended use, known failure modes
- [x] `tests/README.md` — smoke + functional + unit sections, gbert skip policy, coverage table
- [x] `CHANGELOG.md` — entries match actual repo history; version language corrected
- [x] `scripts/train.py` docstring — matches actual implementation (no dead flags, correct regressor logic)
- [x] `conftest.py` docstring — canonical install path documented
- [x] `src/repair_order/features.py` docstring — clean (no duplicate sentences)
- [x] `src/repair_order/pipeline.py` — no unused imports

### Setup
- [x] `pyproject.toml` — version `2.0.0`; all core deps declared; `[dev]` includes `pytest` + `ruff`; pytest config set
- [x] `requirements.txt` — core deps only; gbert extras removed with comments
- [x] Editable install works: `pip install -e .[dev]`
- [x] `Makefile` — POSIX-compliant; targets: `setup`, `data`, `train`, `test`, `predict`, `all`

### Tests
- [x] `tests/test_unit.py` — 10 unit tests (no model required) — **PASS**
- [x] `tests/test_smoke.py` — pipeline integrity tests — **PASS** (8/8, gbert skipped as expected)
- [x] `tests/test_functional.py` — full synthetic-data E2E — **PASS**
- [x] Total: **18 passed, 4 skipped** — Exit code 0

### CI
- [x] Lint step: `ruff check src/ scripts/ tests/` — **passing**
- [x] Import check — **passing**
- [x] Full synthetic-data workflow (generate → train → predict) — **passing**
- [x] Matrix: Python `3.11` + `3.12`

### Synthetic-Data Reproducibility
- [x] `scripts/generate_synthetic_data.py` — generates 500 synthetic orders, 10+ samples/target
- [x] `scripts/train.py --data data/synthetic_orders.json` — trains successfully on synthetic data
- [x] `scripts/predict.py --batch 10` — runs inference and outputs explainable report
- [x] CI runs this exact sequence on every push

### Repo Presentation
- [x] CI badge on README — live and green
- [x] License badge on README — MIT
- [x] No stale unreferenced files at repo root
- [ ] GitHub About text — **needs manual paste in GitHub UI** (text below)
- [ ] GitHub topics — **needs manual paste in GitHub UI** (tags below)

---

## GitHub About Content (paste in GitHub UI)

**One-line description:**
> Two-stage ML pipeline predicting 14 automotive repair work-step durations from unstructured German repair orders.

**About text (160 chars):**
> Two-stage ML pipeline predicting 14 automotive repair work-step durations from unstructured German repair orders. Reproducible on synthetic data. Python 3.11/3.12.

**Topics / Tags:**
```
machine-learning  lightgbm  scikit-learn  nlp  tfidf  python
two-stage-pipeline  automotive  multi-label-classification  repair-orders
```

---

## Release Note Draft (v2.0.0)

> **v2.0.0 — Portfolio-Ready Engineering Pass**
>
> This release restructures the repository for public reviewability and long-term maintainability.
>
> **Highlights:**
> - Full synthetic-data reproducibility: `generate → train → predict` verified on Python 3.11 + 3.12 via CI
> - 10 new deterministic unit tests covering the feature-engineering core (`preprocess_positions`, `build_order_text`, `build_numeric_features`)
> - `ruff` lint gate added to CI and dev extras — enforced on every push
> - All public-facing documentation aligned with the actual implementation
> - `MODEL_CARD.md` now includes the actual metric results table (Macro F1 **0.838**, Freq-Weighted F1 **0.935**, MAE **0.96 hrs**)
> - `pyproject.toml` is now the canonical install source (`pip install -e .[dev]`)
>
> No core model behavior was changed. All original business metrics are unchanged.

---

## State of the Repo

### What Is Now Strong
| Area | Status |
|---|---|
| Public reproducibility | Full: generate → train → predict verified in CI on every push |
| Documentation accuracy | All docs match the actual code (no dead flags, no contradictions) |
| Test coverage | Smoke + unit + functional; feature-engineering core is unit-tested |
| CI | Lint + import check + full pipeline + pytest, Python 3.11 + 3.12 |
| Developer tooling | `pip install -e .[dev]` → pytest + ruff immediately available |
| Repo presentation | Clean structure, confident README, self-contained MODEL_CARD |

### What Remains Optional
| Item | Notes |
|---|---|
| Python 3.10 CI | `requires-python >=3.10` but CI tests 3.11/3.12 only |
| Windows CI job | README documents Windows; CI only tests Ubuntu |
| Type annotations | `features.py`, `pipeline.py` have none |
| `CONTRIBUTING.md` | Helpful for open-source; optional for portfolio |
| Pre-commit hook | ruff could be a pre-commit hook for dev ergonomics |

### Next Engineering Step (if continued)
Add `mypy` or `pyright` type checking in CI with a `[tool.mypy]` config targeting `src/repair_order/` — this would be the most visible remaining engineering signal for a senior reviewer.
