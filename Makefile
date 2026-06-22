.PHONY: install data train predict test serve clean

VENV := .venv
PYTHON := $(VENV)/Scripts/python.exe
PIP   := $(VENV)/Scripts/pip.exe

# --- Install ---------------------------------------------------------------
install: ## Install project in editable mode + dev deps
	$(PIP) install -e ".[api,dev]"

# --- Data -------------------------------------------------------------------
data: ## Regenerate the enriched dataset (20k rows)
	$(PYTHON) -m aharamitra.build_dataset

# --- Training --------------------------------------------------------------
train: ## Build dataset + full benchmark + Optuna tuning + evaluate
	$(PYTHON) -m aharamitra.train

train-quick: ## Train with defaults (skip benchmark + tuning) for iteration speed
	$(PYTHON) -m aharamitra.train --skip-benchmark --skip-tuning

# --- Predict ----------------------------------------------------------------
predict: ## Run a sample prediction using saved models
	$(PYTHON) -m aharamitra.train predict

# --- Tests ------------------------------------------------------------------
test: ## Run pytest
	$(PYTHON) -m pytest tests/ -ra -q

# --- Serve ------------------------------------------------------------------
serve: ## Start FastAPI server (uvicorn)
	$(PYTHON) -m uvicorn api.main:app --reload --port 8000

# --- Clean ------------------------------------------------------------------
clean: ## Remove generated artifacts
	rm -rf models/*.joblib models/*.json
	rm -rf reports/*.png reports/*.csv
	rm -rf data/processed/*.csv
