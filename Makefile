# =============================================================================
# NBA Playoff Predictor — Pipeline Makefile
# =============================================================================
# Usage:
#   make env          - Create conda environment
#   make fetch        - Fetch all raw data (run once; takes a while)
#   make fetch-smoke  - Fetch just 2 seasons for smoke testing
#   make process      - Build processed feature parquet files
#   make validate     - Run pandera schema checks on processed data
#   make eda          - Execute all EDA notebooks
#   make train        - Train all models (logged to MLflow)
#   make evaluate     - Run backtesting + evaluation notebook
#   make predict      - Generate 2026 bracket predictions
#   make test         - Run unit tests
#   make smoke-test   - Run smoke test (fetch 2 seasons + full pipeline)
#   make mlflow-ui    - Launch MLflow UI at http://localhost:5000
#   make clean        - Remove processed files (raw data preserved)
#   make clean-all    - Remove processed + raw data
# =============================================================================

.DEFAULT_GOAL := help

PYTHON    := conda run -n nba-predictor python
PYTEST    := conda run -n nba-predictor pytest
JUPYTER   := conda run -n nba-predictor jupyter nbconvert --to notebook --execute --inplace
MLFLOW    := conda run -n nba-predictor mlflow

PROJECT_ROOT := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
SRC_DIR      := $(PROJECT_ROOT)/src
NB_DIR       := $(PROJECT_ROOT)/notebooks

# ── Environment ───────────────────────────────────────────────────────────────
.PHONY: env
env:
	conda env create -f environment.yml
	conda run -n nba-predictor pip install -e .
	conda run -n nba-predictor pre-commit install
	@echo "Environment ready. Activate with: conda activate nba-predictor"

.PHONY: env-update
env-update:
	conda env update -f environment.yml --prune

# ── Data fetching ─────────────────────────────────────────────────────────────
.PHONY: fetch
fetch:
	@echo "==> Fetching Basketball Reference data (1984–2025)..."
	$(PYTHON) -m nba_predictor.data.bball_ref_scraper \
	    --start-season 1984 \
	    --end-season 2025
	@echo "==> Fetching NBA Stats API data (1996–2025)..."
	$(PYTHON) -m nba_predictor.data.nba_api_fetcher \
	    --start-season 1996 \
	    --end-season 2025
	@echo "==> Data fetch complete."

.PHONY: fetch-smoke
fetch-smoke:
	@echo "==> Smoke-test fetch: 2023–2025 only..."
	$(PYTHON) -m nba_predictor.data.bball_ref_scraper \
	    --start-season 2023 \
	    --end-season 2025
	$(PYTHON) -m nba_predictor.data.nba_api_fetcher \
	    --start-season 2023 \
	    --end-season 2025

# ── Feature processing ────────────────────────────────────────────────────────
.PHONY: process
process:
	@echo "==> Building feature parquet files..."
	$(PYTHON) -m nba_predictor.features.feature_pipeline

.PHONY: validate
validate:
	@echo "==> Running pandera schema validation..."
	$(PYTHON) -m nba_predictor.data.data_validator

# ── EDA notebooks ─────────────────────────────────────────────────────────────
.PHONY: eda
eda:
	@echo "==> Executing EDA notebooks..."
	$(JUPYTER) $(NB_DIR)/00_data_audit.ipynb
	$(JUPYTER) $(NB_DIR)/01_eda_team_stats.ipynb
	$(JUPYTER) $(NB_DIR)/02_eda_player_stats.ipynb
	$(JUPYTER) $(NB_DIR)/03_eda_injuries.ipynb
	$(JUPYTER) $(NB_DIR)/04_feature_engineering_validation.ipynb
	$(JUPYTER) $(NB_DIR)/05_era_normalization.ipynb

# ── Model training ────────────────────────────────────────────────────────────
.PHONY: train
train:
	@echo "==> Training baseline (logistic regression)..."
	$(PYTHON) -m nba_predictor.models.baseline
	@echo "==> Training XGBoost..."
	$(PYTHON) -m nba_predictor.models.gradient_boosting --model xgboost
	@echo "==> Training LightGBM..."
	$(PYTHON) -m nba_predictor.models.gradient_boosting --model lightgbm
	@echo "==> Training stacking ensemble..."
	$(PYTHON) -m nba_predictor.models.ensemble
	@echo "==> Training series length model..."
	$(PYTHON) -m nba_predictor.models.series_length
	@echo "==> All models trained and logged to MLflow."

# ── Evaluation ────────────────────────────────────────────────────────────────
.PHONY: evaluate
evaluate:
	@echo "==> Running backtesting..."
	$(PYTHON) -m nba_predictor.evaluation.backtesting
	@echo "==> Executing evaluation notebooks..."
	$(JUPYTER) $(NB_DIR)/06_modeling_baseline.ipynb
	$(JUPYTER) $(NB_DIR)/07_modeling_advanced.ipynb
	$(JUPYTER) $(NB_DIR)/08_model_evaluation.ipynb

# ── 2026 bracket prediction ───────────────────────────────────────────────────
.PHONY: predict
predict:
	@echo "==> Generating 2026 bracket predictions..."
	$(PYTHON) -m nba_predictor.predict.bracket_simulator \
	    --season 2026
	$(JUPYTER) $(NB_DIR)/09_bracket_prediction_2026.ipynb
	@echo "==> Predictions written to data/predictions/2026/bracket_output.csv"

# ── Testing ───────────────────────────────────────────────────────────────────
.PHONY: test
test:
	$(PYTEST) tests/ -v --tb=short

.PHONY: diagnose-scraper
diagnose-scraper:
	@echo "==> Diagnosing scraper endpoints (1 season each, live network calls)..."
	$(PYTHON) -m nba_predictor.data.diagnose_scraper

.PHONY: test-coverage
test-coverage:
	$(PYTEST) tests/ --cov=nba_predictor --cov-report=html --cov-report=term-missing

.PHONY: smoke-test
smoke-test: fetch-smoke process validate test
	@echo "==> Smoke test complete."

# ── MLflow UI ─────────────────────────────────────────────────────────────────
.PHONY: mlflow-ui
mlflow-ui:
	$(MLFLOW) ui --backend-store-uri mlruns/ --port 5000

# ── Lint / type check ─────────────────────────────────────────────────────────
.PHONY: lint
lint:
	conda run -n nba-predictor ruff check src/ tests/
	conda run -n nba-predictor ruff format --check src/ tests/

.PHONY: format
format:
	conda run -n nba-predictor ruff check --fix src/ tests/
	conda run -n nba-predictor ruff format src/ tests/

# ── Cleanup ───────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	@echo "==> Removing processed files (raw data preserved)..."
	find data/processed -name "*.parquet" -delete
	find data/processed -name "checksums.txt" -delete
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	@echo "==> Clean complete."

.PHONY: clean-all
clean-all: clean
	@echo "==> WARNING: Removing all raw data. Re-run 'make fetch' to restore."
	rm -rf data/raw/bball_ref/team_stats/*
	rm -rf data/raw/bball_ref/player_stats/*
	rm -rf data/raw/bball_ref/playoff_series/*
	rm -rf data/raw/nba_api/*
	rm -rf data/raw/kaggle/*

# ── Full pipeline ─────────────────────────────────────────────────────────────
.PHONY: all
all: fetch process validate train evaluate predict
	@echo "==> Full pipeline complete."

# ── Help ──────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "NBA Playoff Predictor — Makefile targets:"
	@echo ""
	@echo "  make env           Create conda environment"
	@echo "  make fetch         Fetch all raw data (1984–2025)"
	@echo "  make fetch-smoke   Fetch 2 seasons for smoke testing"
	@echo "  make process       Build feature parquet files"
	@echo "  make validate      Run schema checks on processed data"
	@echo "  make eda           Execute EDA notebooks"
	@echo "  make train         Train all models (logged to MLflow)"
	@echo "  make evaluate      Run backtesting + evaluation notebooks"
	@echo "  make predict       Generate 2026 bracket predictions"
	@echo "  make test          Run unit tests"
	@echo "  make diagnose-scraper  Live endpoint validation (columns, row counts)"
	@echo "  make smoke-test    Fetch 2 seasons + full pipeline test"
	@echo "  make mlflow-ui     Launch MLflow UI at http://localhost:5000"
	@echo "  make lint          Run ruff linting checks"
	@echo "  make format        Auto-format with ruff"
	@echo "  make clean         Remove processed files"
	@echo "  make clean-all     Remove processed + raw data"
	@echo "  make all           Run full pipeline end-to-end"
	@echo ""
