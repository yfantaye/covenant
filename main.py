import argparse
import logging
import yaml
import modelv1 
import modelv2 

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main entry point to run the ML pipeline."""
    parser = argparse.ArgumentParser(description="Corporate Failure Prediction Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config_local.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="scottv1",
        choices={**modelv1.MODEL_MAPPING, **modelv2.MODEL_MAPPING}.keys(),
        help="The type of model to train.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "tune"],
        help="Operation mode: 'train' a model or 'tune' hyperparameters.",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run in test mode using data from the test directory specified in config.",
    )
    args = parser.parse_args()

    # --- Load Configuration ---
    config = load_config(args.config)
    config['use_sample_data'] = args.test_run
    config['run_mode'] = args.mode
    config['model_type'] = args.model_type

    if args.test_run:
        assert config['price_date_format'] == "%m/%d/%Y", "Price date format must be %m/%d/%Y"
        assert config['signal_date_format'] == "%m/%d/%Y", "Date format must be %m/%d/%Y"
        assert config['index_date_format'] == "%Y-%m-%d", "Index date format must be %Y-%m-%d"
    else:
        assert config['price_date_format'] == "%Y-%m-%d", "Price date format must be %Y-%m-%d"
        assert config['signal_date_format'] == "%Y-%m-%d", "Signal Date format must be %Y-%m-%d"
        assert config['index_date_format'] == "%Y-%m-%d", "Index date format must be %Y-%m-%d"

    if args.model_type in modelv1.MODEL_MAPPING:
        modelv1.run(config)
    elif args.model_type in modelv2.MODEL_MAPPING:
        modelv2.run(config)
    else:
        raise ValueError(f"Model type '{args.model_type}' not found in MODEL_MAPPING.")



if __name__ == "__main__":
    main()
