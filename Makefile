# Makefile — task runner for work-step-time-prediction
#
# Requires GNU Make and an ACTIVE virtual environment.
# Activate first:
#   Linux/macOS : source .venv/bin/activate
#   Windows     : .\.venv\Scripts\Activate.ps1
#
# If you don't have GNU Make on Windows, use the explicit commands
# documented in README.md Quick Start instead.

.PHONY: help setup data train test predict serve test-api monitor \
        train-challenger promote promote-dry \
        docker-build docker-serve docker-down all

.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "  work-step-time-prediction — available Make targets"
	@echo "  ─────────────────────────────────────────────────────"
	@echo "  make setup           Create .venv and install core + dev deps"
	@echo "  make data            Generate 500-order synthetic dataset"
	@echo "  make train           Train the champion pipeline (models/)"
	@echo "  make test            Run the full pytest suite (-vv)"
	@echo "  make predict         Run batch inference on 10 synthetic orders"
	@echo "  make serve           Start the FastAPI inference service (reload mode)"
	@echo "  make test-api        Run API integration tests only"
	@echo "  make monitor         Generate prediction quality monitoring snapshot"
	@echo "  make train-challenger  Train a challenger (models/challenger/)"
	@echo "  make promote-dry     Evaluate promotion rules without changing files"
	@echo "  make promote         Promote challenger to champion if rules pass"
	@echo "  make docker-build    Build the API container image"
	@echo "  make docker-serve    Build + run the API container (mounts models/)"
	@echo "  make docker-down     Stop and remove the API container"
	@echo "  make all             data → train → test → predict (full workflow)"
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

serve:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test-api:
	pytest tests/test_api.py -vv

monitor:
	python scripts/monitoring_report.py --generate --n 30 --out docs/markdowns/monitoring_snapshot.md

# ── Champion-challenger lifecycle ────────────────────────────────────────────────
train-challenger:
	python scripts/train.py --data data/synthetic_orders.json \
		--target-dir models/challenger

promote-dry:
	python scripts/promote.py --dry-run

promote:
	python scripts/promote.py

# ── Docker targets ────────────────────────────────────────────────────────────
docker-build:
	docker build -t work-step-time-prediction:latest .

docker-serve:
	docker compose up --build

docker-down:
	docker compose down

# ── Full end-to-end workflow ─────────────────────────────────────────────────
all: data train test predict
