# Makefile — task runner for work-step-time-prediction
#
# Requires GNU Make and an ACTIVE virtual environment.
# Activate first:
#   Linux/macOS : source .venv/bin/activate
#   Windows     : .\.venv\Scripts\Activate.ps1
#
# If you don't have GNU Make on Windows, use the explicit commands
# documented in README.md Quick Start instead.

.PHONY: help setup data train test predict all

.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "  work-step-time-prediction — available Make targets"
	@echo "  ─────────────────────────────────────────────────────"
	@echo "  make setup    Create .venv and install core + dev deps"
	@echo "  make data     Generate 500-order synthetic dataset"
	@echo "  make train    Train the two-stage pipeline (synthetic data)"
	@echo "  make test     Run the full pytest suite (-vv)"
	@echo "  make predict  Run batch inference on 10 synthetic orders"
	@echo "  make all      data → train → test → predict (full workflow)"
	@echo ""

# ── Environment setup ────────────────────────────────────────────────────────
setup:
	python -m venv .venv
	python -m pip install --upgrade pip
	pip install -e ".[dev]"
	@echo ""
	@echo "  ✓ Environment ready. Activate it before running other targets:"
	@echo "    Linux/macOS : source .venv/bin/activate"
	@echo "    Windows     : .\.venv\Scripts\Activate.ps1"
	@echo ""

# ── Workflow targets (require active venv) ───────────────────────────────────
data:
	python scripts/generate_synthetic_data.py

train:
	python scripts/train.py --data data/synthetic_orders.json

test:
	pytest -vv

predict:
	python scripts/predict.py --batch 10

# ── Full end-to-end workflow ─────────────────────────────────────────────────
all: data train test predict
