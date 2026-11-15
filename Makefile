# ==============================================================================
# Customer Intelligence Engine (CIE) - Automation Makefile
# ==============================================================================

PYTHON ?= python

.PHONY: help all clean features segment train evaluate verify test lint format

help:
	@echo Customer Intelligence Engine (CIE) Workflow Commands:
	@echo   make clean      - Clean raw Online Retail II Excel data into parquets
	@echo   make features   - Generate windowed features and 90-day future targets
	@echo   make segment    - Train unsupervised RFM segmentation pipeline and export profiles
	@echo   make train      - Train and freeze champion Tweedie GLM pipeline
	@echo   make evaluate   - Score holdout test set and export diagnostic tables
	@echo   make verify     - Assert exact 1e-9 prediction parity against saved golden records
	@echo   make all        - Run end-to-end pipeline from raw data to verification
	@echo   make test       - Run pytest test suite
	@echo   make lint       - Run ruff linter
	@echo   make format     - Run ruff code formatter

clean:
	$(PYTHON) -m cie.pipeline.cli clean

features:
	$(PYTHON) -m cie.pipeline.cli features

segment:
	$(PYTHON) -m cie.pipeline.cli segment

train:
	$(PYTHON) -m cie.pipeline.cli train

evaluate:
	$(PYTHON) -m cie.pipeline.cli evaluate

verify:
	$(PYTHON) -m cie.pipeline.cli verify

all: clean features segment train evaluate verify

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .
