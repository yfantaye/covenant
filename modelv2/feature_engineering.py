import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np

class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    """
    A pipeline to transform raw data dictionary into a final feature matrix.
    """
    def __init__(self, config):
        self.config = config
        self.prediction_horizon_days = config['prediction_horizon_days']
        self.rolling_window_days = config['rolling_window_days']

    def fit(self, data_dict, y=None):
        # This pipeline is stateless during fit, but could learn e.g. scalers
        return self

    def transform(self, data_dict):
        """
        Applies all feature engineering steps.
        """
        # For demonstration, we'll use the binary signals directly and create labels.
        # A full implementation would add all features from the proposal here.
        
        df = data_dict["binary_signals"].copy()
        df_failures = data_dict["failure_list"].copy()

        # --- 1. Create Robust Labels (No Look-ahead Bias) ---
        df_failures['failure_date'] = pd.to_datetime(df_failures['peakYear'])
        failure_dates = df_failures.set_index('companyid')['failure_date']
        
        df['failure_date'] = df['companyid'].map(failure_dates)
        
        #horizon = pd.Timedelta(days=self.prediction_horizon_days)
        
        # The target: Does the company fail within the next year or has failed already?
        df['target'] = (
            df['failure_date'] > df['signal_date']
            #(df['failure_date'] > df['signal_date']) &
            #(df['failure_date'] <= df['signal_date'] + horizon)
        ).astype(int)

        # --- 2. Feature Creation (Example: Rolling Counts) ---
        signal_cols = [c for c in df.columns if c not in ['companyid', 'signal_date', 'failure_date', 'target']]
        
        # Ensure signal columns are numeric
        for col in signal_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df = df.sort_values(by=['companyid', 'signal_date'])
        
        # --- 2.1. Create Rolling Features ---
        rolling_cols = []
        if self.rolling_window_days > 0:
            for col in signal_cols:
                # Rolling sum over 90 days for each company
                df[f'{col}_rolling_90d'] = df.groupby('companyid')[col].transform(
                    lambda x: x.rolling(window=self.rolling_window_days, min_periods=1).sum()
                )
            rolling_cols = [f'{col}_rolling_90d' for col in signal_cols]

        # --- 3. Final Cleanup ---
        # Select feature columns (original signals + new rolling features)
        feature_cols = signal_cols + rolling_cols
        
        final_df = df[['companyid', 'signal_date', 'target'] + feature_cols].dropna()
        
        return final_df
