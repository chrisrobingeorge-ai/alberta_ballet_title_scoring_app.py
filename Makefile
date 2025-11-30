# Alberta Ballet Title Scoring App - Makefile
#
# This Makefile provides convenient commands for running the ML pipeline
# and common development tasks.
#
# Usage:
#   make full-pipeline     # Run complete ML pipeline end-to-end
#   make full-pipeline-tune # Run pipeline with hyperparameter tuning
#   make build-dataset     # Build the modelling dataset only
#   make backtest          # Run backtesting only
#   make train             # Train the model only
#   make test              # Run unit tests
#   make clean             # Clean generated artifacts

.PHONY: full-pipeline full-pipeline-tune build-dataset backtest train test clean help

# Default Python interpreter
PYTHON := python

# Default paths
HISTORY_PATH := data/productions/history_city_sales.csv
BASELINES_PATH := data/productions/baselines.csv
PAST_RUNS_PATH := data/productions/past_runs.csv

# ==============================================================================
# Main Targets
# ==============================================================================

## Run the complete ML pipeline (build dataset, backtest, train model)
full-pipeline:
	$(PYTHON) scripts/run_full_pipeline.py \
		--history $(HISTORY_PATH) \
		--baselines $(BASELINES_PATH) \
		--past-runs $(PAST_RUNS_PATH)

## Run the complete ML pipeline with hyperparameter tuning
full-pipeline-tune:
	$(PYTHON) scripts/run_full_pipeline.py \
		--history $(HISTORY_PATH) \
		--baselines $(BASELINES_PATH) \
		--past-runs $(PAST_RUNS_PATH) \
		--tune

## Run the complete ML pipeline with SHAP explanations
full-pipeline-shap:
	$(PYTHON) scripts/run_full_pipeline.py \
		--history $(HISTORY_PATH) \
		--baselines $(BASELINES_PATH) \
		--past-runs $(PAST_RUNS_PATH) \
		--save-shap

## Run the complete ML pipeline with tuning and SHAP
full-pipeline-all:
	$(PYTHON) scripts/run_full_pipeline.py \
		--history $(HISTORY_PATH) \
		--baselines $(BASELINES_PATH) \
		--past-runs $(PAST_RUNS_PATH) \
		--tune --save-shap

# ==============================================================================
# Individual Pipeline Steps
# ==============================================================================

## Build the leak-free modelling dataset
build-dataset:
	$(PYTHON) scripts/build_modelling_dataset.py \
		--history $(HISTORY_PATH) \
		--baselines $(BASELINES_PATH) \
		--past-runs $(PAST_RUNS_PATH)

## Run time-aware backtesting
backtest:
	$(PYTHON) scripts/backtest_timeaware.py

## Train the safe model
train:
	$(PYTHON) scripts/train_safe_model.py

## Train with hyperparameter tuning
train-tune:
	$(PYTHON) scripts/train_safe_model.py --tune

## Train with SHAP explanations
train-shap:
	$(PYTHON) scripts/train_safe_model.py --save-shap

# ==============================================================================
# Development & Testing
# ==============================================================================

## Run all unit tests
test:
	$(PYTHON) -m pytest tests/ -v

## Run pipeline smoke tests
test-pipeline:
	$(PYTHON) -m pytest tests/test_full_pipeline.py -v

## Run end-to-end pipeline tests
test-e2e:
	$(PYTHON) -m pytest tests/test_end_to_end_pipeline.py -v

## Run the Streamlit app
run:
	streamlit run streamlit_app.py

# ==============================================================================
# Cleanup
# ==============================================================================

## Clean generated artifacts (keeps models and important results)
clean:
	rm -f data/modelling_dataset.csv
	rm -f diagnostics/modelling_dataset_report.json
	rm -rf results/*/
	rm -rf __pycache__/
	rm -rf tests/__pycache__/
	rm -rf scripts/__pycache__/
	rm -rf .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

## Deep clean (removes models and all results - use with caution)
clean-all: clean
	rm -f models/*.joblib
	rm -f results/*.csv
	rm -f results/*.json
	rm -rf results/plots/
	rm -rf results/shap/

# ==============================================================================
# Help
# ==============================================================================

## Show this help message
help:
	@echo ""
	@echo "Alberta Ballet Title Scoring App - Available Commands"
	@echo "====================================================="
	@echo ""
	@echo "Pipeline Commands:"
	@echo "  make full-pipeline       Run complete ML pipeline end-to-end"
	@echo "  make full-pipeline-tune  Run pipeline with hyperparameter tuning"
	@echo "  make full-pipeline-shap  Run pipeline with SHAP explanations"
	@echo "  make full-pipeline-all   Run pipeline with tuning and SHAP"
	@echo ""
	@echo "Individual Steps:"
	@echo "  make build-dataset       Build the modelling dataset"
	@echo "  make backtest            Run time-aware backtesting"
	@echo "  make train               Train the safe model"
	@echo "  make train-tune          Train with hyperparameter tuning"
	@echo ""
	@echo "Development:"
	@echo "  make test                Run all unit tests"
	@echo "  make test-pipeline       Run pipeline smoke tests"
	@echo "  make run                 Start the Streamlit app"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean               Clean generated artifacts"
	@echo "  make clean-all           Deep clean (removes models too)"
	@echo ""

# Default target
.DEFAULT_GOAL := help
