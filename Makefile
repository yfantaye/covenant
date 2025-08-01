# Makefile for Corporate Failure Prediction Project

# --- Variables ---
VENV_NAME := .venv
PYTHON := $(VENV_NAME)/bin/python
UV := uv
DATA_DIR := ./data
TEST_DATA_DIR := ./tests/test_data
MODEL_OUTPUT_DIR := ./model_outputs

.PHONY: all setup clean test-run run-prod tune

all: setup

# --- Environment Setup ---
setup: $(VENV_NAME)/bin/activate

$(VENV_NAME)/bin/activate: requirements.txt
	@echo ">>> Setting up virtual environment..."
	$(UV) venv $(VENV_NAME)
	$(UV) pip install -r requirements.txt --python $(PYTHON)
	@echo ">>> Setup complete. Activate with: source $(VENV_NAME)/bin/activate"

# --- Cleaning ---
clean:
	@echo ">>> Cleaning up project..."
	rm -rf $(VENV_NAME)
	rm -rf $(MODEL_OUTPUT_DIR)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo ">>> Cleanup complete."

# --- Pipeline Execution ---

# Run a quick test on sample data
test-run:
	@echo ">>> Running a test pipeline on sample data..."
	$(PYTHON) main.py \
		--data-dir $(TEST_DATA_DIR) \
		--model-type lightgbm \
		--mode train
	@echo ">>> Test run finished."

# Run a full production training run
# Usage: make run-prod MODEL=<model_type>
# Example: make run-prod MODEL=lightgbm
run-prod:
	@echo ">>> Running production training for model: $(MODEL)..."
	$(PYTHON) main.py \
		--data-dir $(DATA_DIR) \
		--model-type $(MODEL) \
		--mode train
	@echo ">>> Production run for $(MODEL) finished."

# Run hyperparameter tuning
# Usage: make tune MODEL=<model_type>
# Example: make tune MODEL=lightgbm
tune:
	@echo ">>> Running hyperparameter tuning for model: $(MODEL)..."
	$(PYTHON) main.py \
		--data-dir $(DATA_DIR) \
		--model-type $(MODEL) \
		--mode tune
	@echo ">>> Tuning for $(MODEL) finished."

