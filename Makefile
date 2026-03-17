.PHONY: help setup test train predict
.DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@echo "  make setup    - Create virtual environment and install dependencies"
	@echo "  make test     - Run pytest suite"
	@echo "  make train    - Run the core training pipeline (Stage 1 & Stage 2)"
	@echo "  make predict  - Run inference using the saved pipeline"

setup:
	python -m venv .venv
	.venv/Scripts/python -m pip install --upgrade pip
	.venv/Scripts/python -m pip install -e ".[dev]"

test:
	.venv/Scripts/pytest tests/ -v

train:
	.venv/Scripts/python scripts/train.py

predict:
	.venv/Scripts/python scripts/predict.py --batch 10
