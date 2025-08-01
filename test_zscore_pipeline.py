#!/usr/bin/env python3
"""
Test script for ZScorePipeline to verify it works correctly.
"""

import pandas as pd
import numpy as np
import logging
from modelv2.model_pipeline import ZScorePipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_sample_data():
    """Create sample data for testing the ZScorePipeline."""
    np.random.seed(42)
    
    # Create sample data with 100 companies, 50 time periods each
    n_companies = 100
    n_periods = 50
    n_features = 20
    
    data = []
    for company_id in range(n_companies):
        for period in range(n_periods):
            row = {
                'companyid': f'company_{company_id}',
                'signal_date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=period),
                'target': 0  # Default to non-failure
            }
            
            # Add binary features
            for i in range(n_features):
                row[f'feature_{i}'] = np.random.randint(0, 2)
            
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Create some failure patterns
    # Make some features more predictive of failure
    failure_companies = np.random.choice(n_companies, size=20, replace=False)
    
    for company_id in failure_companies:
        company_mask = df['companyid'] == f'company_{company_id}'
        
        # Make some features more likely to be 1 for failed companies
        df.loc[company_mask, 'feature_0'] = np.random.choice([0, 1], size=sum(company_mask), p=[0.2, 0.8])
        df.loc[company_mask, 'feature_1'] = np.random.choice([0, 1], size=sum(company_mask), p=[0.1, 0.9])
        
        # Mark these companies as failures
        df.loc[company_mask, 'target'] = 1
    
    return df

def test_zscore_pipeline():
    """Test the ZScorePipeline with sample data."""
    print("=== Testing ZScorePipeline ===")
    
    # Create sample data
    df = create_sample_data()
    print(f"Sample data shape: {df.shape}")
    print(f"Failure rate: {df['target'].mean():.3f}")
    
    # Initialize pipeline
    pipeline = ZScorePipeline()
    
    # Run the pipeline
    print("\nRunning pipeline...")
    results = pipeline.run(df)
    
    # Print results
    print(f"\nEvaluation results:")
    print(f"ROC AUC: {results['metrics']['roc_auc']:.4f}")
    print(f"PR AUC: {results['metrics']['pr_auc']:.4f}")
    
    # Save artifacts
    pipeline.save_artifacts()
    print(f"\nArtifacts saved to: {pipeline.output_dir}")
    
    # Print top features
    if pipeline.weights is not None:
        print(f"\nTop 10 feature weights:")
        top_weights = pipeline.weights.sort_values(ascending=False).head(10)
        for feature, weight in top_weights.items():
            print(f"  {feature}: {weight:.4f}")
    
    return results

if __name__ == "__main__":
    test_zscore_pipeline() 