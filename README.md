# Corporate Failure Prediction Model

This project provides a modular and extensible Python framework for training, evaluating, and deploying machine learning models to predict corporate failure. It is designed to replace a legacy system with a robust, statistically sound, and highly explainable MLOps pipeline.

## Project Philosophy

The core philosophy is to build a system that is:

- **Modular:** Each logical component (data ingestion, feature engineering, modeling, evaluation) is separated into its own module for clarity and maintainability.
- **Reusable:** A base `ModelPipeline` class defines a common interface, allowing new models to be integrated with minimal effort.
- **Robustly Validated:** The framework enforces a strict, time-based data splitting and validation strategy to prevent look-ahead bias and ensure models generalize to unseen data.
- **Automated:** A `Makefile` provides simple commands to set up the environment, install dependencies, and run training/prediction pipelines.
- **Tunable & Trackable:** Integrated support for hyperparameter tuning with Optuna and experiment tracking with MLflow (optional, hooks included).

## Project Structure

```
corp-failure-predictor/
├── data/                      # Main data folder for production runs
│   ├── binarySignalsPart01.csv
│   └── ...
├── tests/                     # Test data and scripts
│   ├── test_data/             # Sample data for test runs
│   │   ├── binarySignalsPart_sample.csv
│   │   └── ...
│   └── test_pipeline.py
├── corp_failure_predictor/    # Main source code package
│   ├── __init__.py
│   ├── data_ingestion.py      # Loads and merges all data sources.
│   ├── feature_engineering.py # Creates a rich feature set from raw data.
│   ├── model_pipeline.py      # Base model class and specific model implementations.
│   ├── tuning.py              # Hyperparameter tuning using Optuna.
│   └── evaluation.py          # Calculates and plots performance metrics.
├── main.py                    # Main entry point to run pipelines.
├── Makefile                   # Automation for setup, testing, and runs.
├── requirements.txt           # Project dependencies.
└── README.md                  # This file.
```

## Setup and Installation

This project uses `uv` for fast package management and virtual environment creation.

1.  **Prerequisites:** Ensure you have Python 3.9+ and `uv` installed.
    ```bash
    pip install uv
    ```

2.  **Create Virtual Environment & Install Dependencies:** The `Makefile` automates this. From the project root, run:
    ```bash
    make setup
    ```
    This command creates a `.venv` directory, activates it, and installs all packages from `requirements.txt`.

## How to Run the Pipeline

The `main.py` script is the primary entry point. It is controlled via command-line arguments. The `Makefile` provides convenient shortcuts.

### Running a Test Pipeline

This uses the small sample data located in `tests/test_data/` to quickly verify that the entire pipeline runs without errors.

```bash
make test-run
```

This is equivalent to running:
```bash
.venv/bin/python main.py --data-dir ./tests/test_data/ --model-type lightgbm --mode train
```

### Running a Full Production Training Pipeline

This uses the complete dataset in the `data/` directory.

1.  **Train a Logistic Regression Baseline:**
    ```bash
    make run-prod MODEL=logistic_regression
    ```

2.  **Train a LightGBM Model:**
    ```bash
    make run-prod MODEL=lightgbm
    ```

3.  **Train a Cox Proportional Hazards Model:**
    ```bash
    make run-prod MODEL=coxph
    ```

### Running Hyperparameter Tuning

To find the best parameters for the LightGBM model using Optuna:

```bash
make tune MODEL=lightgbm
```

This will run a study and output the best hyperparameters found on the validation set.

## Model Development

To add a new model (e.g., `MyNewModel`):

1.  Open `corp_failure_predictor/model_pipeline.py`.
2.  Create a new class `MyNewModelPipeline` that inherits from `BaseModelPipeline`.
3.  Implement the `_train`, `_predict`, and `_evaluate` methods for your new model.
4.  Add your new model to the `MODEL_MAPPING` dictionary in `main.py`.
5.  You can now run your model using `make run-prod MODEL=MyNewModel`.
