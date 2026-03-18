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
	# Detect OS and set executable path
	ifeq ($(OS),Windows_NT)
		VENV_BIN=.venv\\Scripts
	else
		VENV_BIN=.venv/bin
	endif
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/python -m pip install -e "[dev]"

test:
	$(VENV_BIN)/pytest tests -v

train:
	$(VENV_BIN)/python scripts/train.py

predict:
	$(VENV_BIN)/python scripts/predict.py --batch 10
