import unittest
import os
import pandas as pd
import yaml
from modelv2.data_ingestion import load_all_data
from modelv2.feature_engineering import FeatureEngineeringPipeline
from modelv2.model_pipeline import LightGBMPipeline

class TestFullPipeline(unittest.TestCase):
    """
    A simple integration test to ensure the full pipeline runs on sample data.
    """
    def setUp(self):
        """Set up for the test."""
        self.config_path = 'config.yaml'
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.test_data_dir = self.config['test_data_directory']
        self.assertTrue(os.path.exists(self.test_data_dir), f"Test data directory not found at {self.test_data_dir}")

    def test_pipeline_run(self):
        """
        Tests a full run of the data ingestion, feature engineering, and a model pipeline.
        """
        # 1. Data Ingestion
        raw_data = load_all_data(self.test_data_dir)
        self.assertIn("binary_signals", raw_data)
        self.assertFalse(raw_data["binary_signals"].empty)

        # 2. Feature Engineering
        feature_pipeline = FeatureEngineeringPipeline(config=self.config['feature_engineering'])
        processed_df = feature_pipeline.fit_transform(raw_data)
        self.assertIsInstance(processed_df, pd.DataFrame)
        self.assertIn("target", processed_df.columns)
        self.assertIn("duration", processed_df.columns)

        # 3. Model Pipeline
        model_params = self.config['models'].get('lightgbm', {})
        # Use fewer estimators for a quick test run
        model_params['n_estimators'] = 10
        
        pipeline = LightGBMPipeline(params=model_params, config=self.config)
        results = pipeline.run(processed_df)

        # 4. Check Results
        self.assertIn('metrics', results)
        self.assertIn('roc_auc', results['metrics'])
        self.assertGreater(results['metrics']['roc_auc'], 0.0) # Should be better than random

if __name__ == '__main__':
    unittest.main()
