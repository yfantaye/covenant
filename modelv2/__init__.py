from .data_ingestion import load_all_data
from .feature_engineering import FeatureEngineeringPipeline
from .model_pipeline import (
    LightGBMPipeline,
    LogisticRegressionPipeline,
    CoxPHPipeline,
    ZScorePipeline,
)
from .tuning import OptunaTuner
import logging

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

MODEL_MAPPING = {
    "lightgbm": LightGBMPipeline,
    "logistic_regression": LogisticRegressionPipeline,
    "coxph": CoxPHPipeline,
    "z_score": ZScorePipeline,
}

def run(config):
    '''
    Runs the model pipeline.
    '''

    test_run = config['use_sample_data']    
    run_mode = config['run_mode']
    model_type = config['model_type']
    model_pipeline_class = MODEL_MAPPING[model_type]

    if test_run:
        data_dir = config['test_data_directory'] 
    else: 
        data_dir = config['data_directory']

    
    if test_run:
        logging.info(">>> RUNNING USING TEST DATA <<<")
    else:
        logging.info(">>> RUNNING USING PRODUCTION DATA <<<")

    logging.info(f"--- Starting Pipeline in '{run_mode}' mode for model '{model_type}' ---")

    # --- Phase 1: Data Ingestion & Feature Engineering ---
    logging.info(f"Loading and merging raw data from: {data_dir}")
    raw_data = load_all_data(data_dir)

    logging.info("Starting feature engineering...")
    feature_pipeline = FeatureEngineeringPipeline(config)
    processed_df = feature_pipeline.fit_transform(raw_data)

    # --- Phase 2: Model Training or Tuning ---
    
    model_params = config['models'].get(model_type, {})
    
    if run_mode == "train":
        logging.info(f"Initializing {model_type} pipeline with default params...")
        pipeline = model_pipeline_class(params=model_params, config=config)

        logging.info("Running training and evaluation pipeline...")
        pipeline.run(processed_df)

    elif run_mode == "tune":
        if model_type not in ["lightgbm"]: # Add other tunable models here
            logging.warning(f"Tuning is not implemented for '{model_type}'. Skipping.")
            return
            
        logging.info(f"Starting hyperparameter tuning for {model_type}...")
        tuner = OptunaTuner(model_pipeline_class, processed_df, config)
        best_params = tuner.tune()
        logging.info(f"Best parameters found: {best_params}")
        logging.info("To use these, update your config.yaml and re-run in 'train' mode.")

    logging.info("--- Pipeline Finished ---")
