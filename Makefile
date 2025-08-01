# Makefile for Corporate Failure Prediction Project

# --- Variables ---
VENV_NAME := .venv
PYTHON := /usr/local/bin/python3
UV := uv
DATA_DIR := ./data
TEST_DATA_DIR := ./tests/test_data
MODEL_OUTPUT_DIR := ./model_outputs
CONFIG_FILE := config_local.yaml

.PHONY: all setup clean test-run run-prod tune install run check-encoding setup-backend

all: setup

# --- Environment Setup ---
setup:
	@echo ">>> Setting up backend environment..."
	@echo ">>> Using Python: $(PYTHON)"
	$(PYTHON) --version
	$(UV) venv $(VENV_NAME) --python $(PYTHON)
	. $(VENV_NAME)/bin/activate && $(UV) pip install -r requirements.txt
	@echo ">>> Backend setup complete!"

# Setup with minimal requirements (no shap)
setup-full:
	@echo ">>> Setting up backend environment with minimal requirements..."
	@echo ">>> Using Python: $(PYTHON)"
	$(PYTHON) --version
	$(UV) venv $(VENV_NAME) --python $(PYTHON)
	. $(VENV_NAME)/bin/activate && $(UV) pip install -r requirements-full.txt
	@echo ">>> Minimal backend setup complete!"



# Check Python version compatibility
check-python:
	@echo ">>> Checking Python version compatibility..."
	$(PYTHON) --version
	@echo ">>> Python version check complete!"

# --- Package Management ---
install:
	@echo ">>> Installing Python package..."
	@if [ -z "$(PACKAGE)" ]; then \
		echo "Usage: make install PACKAGE=<package_name>"; \
		exit 1; \
	fi
	. $(VENV_NAME)/bin/activate && $(UV) pip install $(PACKAGE)
	@echo ">>> Adding package to requirements.txt..."
	. $(VENV_NAME)/bin/activate && $(UV) pip freeze | grep -i $(PACKAGE) >> requirements.txt
	@echo ">>> Package $(PACKAGE) installed and added to requirements.txt"

# --- Command Execution ---
run:
	@echo ">>> Running command in virtual environment..."
	@if [ -z "$(COMMAND)" ]; then \
		echo "Usage: make run COMMAND='<command_to_run>'"; \
		exit 1; \
	fi
	. $(VENV_NAME)/bin/activate && $(COMMAND)

# --- File Encoding Check ---
check-encoding:
	@echo ">>> Checking for files with non-UTF-8 encoding..."
	@find . -type f -name "*.py" -exec file -I {} \; | grep -v "utf-8" || echo "All Python files are UTF-8 encoded"

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
# Usage: make test-run MODEL=<model_type>
# Example: make test-run MODEL=z_score
test-run:
	@echo ">>> Running a test pipeline on sample data..."
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make test-run MODEL=<model_type>"; \
		echo "Available models: lightgbm, logistic_regression, coxph, z_score, scottv1, scottv2"; \
		exit 1; \
	fi
	. $(VENV_NAME)/bin/activate && python main.py \
		--test-run \
		--model-type $(MODEL) \
		--mode train \
		--config $(CONFIG_FILE)
	@echo ">>> Test run finished."

# Run a full production training run
# Usage: make run-prod MODEL=<model_type>
# Example: make run-prod MODEL=lightgbm
run-prod:
	@echo ">>> Running production training for model: $(MODEL)..."
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make run-prod MODEL=<model_type>"; \
		echo "Available models: lightgbm, logistic_regression, coxph, z_score, scottv1, scottv2"; \
		exit 1; \
	fi
	. $(VENV_NAME)/bin/activate && python main.py \
		--model-type $(MODEL) \
		--mode train \
		--config $(CONFIG_FILE)
	@echo ">>> Production run for $(MODEL) finished."

# Run hyperparameter tuning
# Usage: make tune MODEL=<model_type>
# Example: make tune MODEL=lightgbm
tune:
	@echo ">>> Running hyperparameter tuning for model: $(MODEL)..."
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make tune MODEL=<model_type>"; \
		echo "Available models: lightgbm"; \
		exit 1; \
	fi
	. $(VENV_NAME)/bin/activate && python main.py \
		--model-type $(MODEL) \
		--mode tune \
		--config $(CONFIG_FILE)
	@echo ">>> Tuning for $(MODEL) finished."

# --- Quick Commands ---

# Quick test with scottv1 model
test-scottv1:
	@echo ">>> Testing scottv1 model..."
	$(MAKE) test-run MODEL=scottv1

# Quick test with scottv2 model
test-scottv2:
	@echo ">>> Testing scottv2 model..."
	$(MAKE) test-run MODEL=scottv2

# Quick test with scottv3 model
test-scottv3:
	@echo ">>> Testing scottv3 model..."
	$(MAKE) test-run MODEL=scottv3

# Quick test with z_score model
test-zscore:
	@echo ">>> Testing z_score model..."
	$(MAKE) test-run MODEL=z_score

# Quick test with lightgbm model
test-lightgbm:
	@echo ">>> Testing lightgbm model..."
	$(MAKE) test-run MODEL=lightgbm

# Quick test with logistic regression model
test-logistic:
	@echo ">>> Testing logistic regression model..."
	$(MAKE) test-run MODEL=logistic_regression

# Quick test with coxph model
test-coxph:
	@echo ">>> Testing coxph model..."
	$(MAKE) test-run MODEL=coxph

# --- Jupyter Development ---

# Start Jupyter Notebook
jupyter:
	@echo ">>> Starting Jupyter Notebook..."
	@echo ">>> Make sure to select 'Covenant v2 Environment' kernel in your notebooks"
	. $(VENV_NAME)/bin/activate && jupyter notebook --notebook-dir=.

# Start Jupyter Lab
jupyter-lab:
	@echo ">>> Starting Jupyter Lab..."
	@echo ">>> Make sure to select 'Covenant v2 Environment' kernel in your notebooks"
	. $(VENV_NAME)/bin/activate && jupyter lab --notebook-dir=.

# Install Jupyter kernel for this environment
install-kernel:
	@echo ">>> Installing Jupyter kernel for covenantv2 environment..."
	. $(VENV_NAME)/bin/activate && python -m ipykernel install --user --name=covenantv2 --display-name="Covenant v2 Environment"
	@echo ">>> Kernel installed successfully!"
	@echo ">>> Available kernels:"
	. $(VENV_NAME)/bin/activate && jupyter kernelspec list

# List available Jupyter kernels
list-kernels:
	@echo ">>> Available Jupyter kernels:"
	. $(VENV_NAME)/bin/activate && jupyter kernelspec list

# --- Development Helpers ---

# Show available models
models:
	@echo ">>> Available models:"
	@echo "  - lightgbm: Gradient boosting model"
	@echo "  - logistic_regression: Linear model with regularization"
	@echo "  - coxph: Survival analysis model"
	@echo "  - z_score: Z-score based feature weighting model"
	@echo "  - scottv1: Scott version 1 model"
	@echo "  - scottv2: Scott version 2 model"

# Show help
help:
	@echo ">>> Available commands:"
	@echo "  setup: Set up the virtual environment (may fail with shap)"
	@echo "  setup-full: Set up with full requirements (with shap)"
	@echo "  check-python: Check Python version compatibility"
	@echo "  install PACKAGE=<name>: Install a Python package"
	@echo "  run COMMAND='<cmd>': Run a command in the virtual environment"
	@echo "  test-run MODEL=<type>: Run a test with specified model"
	@echo "  run-prod MODEL=<type>: Run production training"
	@echo "  tune MODEL=<type>: Run hyperparameter tuning"
	@echo "  test-scottv1: Quick test with scottv1 model"
	@echo "  test-scottv2: Quick test with scottv2 model"
	@echo "  test-zscore: Quick test with z_score model"
	@echo "  test-lightgbm: Quick test with lightgbm model"
	@echo "  test-logistic: Quick test with logistic regression model"
	@echo "  test-coxph: Quick test with coxph model"
	@echo "  models: Show available models"
	@echo "  clean: Clean up the project"
	@echo "  check-encoding: Check file encodings"
	@echo ""
	@echo ">>> Jupyter Development:"
	@echo "  jupyter: Start Jupyter Notebook"
	@echo "  jupyter-lab: Start Jupyter Lab"
	@echo "  install-kernel: Install Jupyter kernel for this environment"
	@echo "  list-kernels: List available Jupyter kernels"
	@echo ""
	@echo ">>> Python 3.13 Compatibility Notes:"
	@echo "  - Use 'setup-minimal' if the main setup fails"
	@echo "  - Use 'setup-with-shap' for full functionality"
	@echo "  - shap may have compatibility issues with Python 3.13"

