import pandas as pd
import numpy as np
import os
import glob
from datetime import timedelta
import logging

import yfinance as yf
from scipy import stats


# --- Configuration ---
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


class ScottStrategy:
    def __init__(self, config):
        self.config = config
        self.model_name = config['model_type']
        self.data_dir = config.get('data_directory', './data')
        self.output_dir = f"{config.get('model_output_directory', './output')}/{self.model_name}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.use_sample_data = config.get('use_sample_data', False)

        # --- Model Specific Parameters ---
        self.date_format = config.get('modelv1').get('date_format', '%Y-%m-%d')
        self.index_date_format = config.get('index_date_format', '%Y-%m-%d')
        self.price_date_format = config.get('price_date_format', '%d/%m/%d')
        self.signal_date_format = config.get('signal_date_format', '%d/%m/%d')
        
        self.colmap = config.get('modelv1').get('colmap', {})
        self.fail_days = config.get('modelv1').get('fail_days', 180)
        self.trading_days = config.get('modelv1').get('trading_days', 252)
        self.rolling_window_days = config.get('modelv1').get('rolling_window_days', 90)



    def _save_artifacts(self, 
                        scores: pd.DataFrame = None, 
                        weights: pd.Series = None
                        ):
        """Saves the trained model and evaluation results."""
        if scores is not None:
            scores.to_csv(os.path.join(self.output_dir, "scores.csv"))
        if weights is not None:
            weights.to_csv(os.path.join(self.output_dir, "weights.csv"))

    def _find_files(self, pattern: str):
        files = glob.glob(os.path.join(self.data_dir, f'{pattern}*.csv'))
        if not files:
            raise FileNotFoundError(f"No files found for pattern: {pattern} in {self.data_dir}")
        return files

    def _split_data(self, df_labeled_signals: pd.DataFrame, 
                target_col: str='fail') -> tuple:
        """
        Splits the data into training and testing sets.
        """
        signal_cols = [col for col in df_labeled_signals.columns if col not in ['companyid', 'signal_date', target_col]]
        
        failed_group = df_labeled_signals[df_labeled_signals[target_col] == 1]
        non_failed_group = df_labeled_signals[df_labeled_signals[target_col] == 0]

        return signal_cols, failed_group, non_failed_group

    def compute_failure_probability_timeline(self, df: pd.DataFrame, failure_year: int):
        """
        Calculates a daily "Financial Distress Score" based on rolling alpha and beta
        to identify the most probable time that failure became critical.

        Args:
            df: The DataFrame with the stock, index, and risk-free rate data.
            failure_year: The year the company failed.

        Returns:
            tuple: A pandas DataFrame with the daily timeline and the date of the highest distress score.
        """ 

        # Combine, align, and calculate returns
        df['stock_return'] = np.log(df['stock'] / df['stock'].shift(1))
        df['index_return'] = np.log(df['index'] / df['index'].shift(1))
        df = df.dropna()  

        # Calculate rolling alpha and beta (90-day window is common)
        window = self.rolling_window_days
        rolling_results = []

        for i in range(len(df) - window):
            subset = df.iloc[i:i+window]
            
            # Calculate beta for the window
            slope, _, _, _, _ = stats.linregress(x=subset['index_return'], 
                                                y=subset['stock_return'])
            beta = slope
            
            # Calculate alpha for the window (annualized)
            trading_days = self.trading_days
            annual_stock_return = subset['stock_return'].mean() * trading_days
            annual_index_return = subset['index_return'].mean() * trading_days
            annual_risk_free_rate = subset['risk_free_rate'].mean() / 100
            expected_return = annual_risk_free_rate + beta * (annual_index_return - annual_risk_free_rate)
            alpha = annual_stock_return - expected_return
            
            rolling_results.append({
                'date': df.index[i + window],
                'alpha': alpha,
                'beta': beta
            })

        if not rolling_results:
            print("Could not generate rolling results. Check data availability.")
            return None, None
            
        timeline_df = pd.DataFrame(rolling_results).set_index('date')
        
        # Filter for the actual failure year
        timeline_df = timeline_df[timeline_df.index.year == failure_year]

        # Calculate Financial Distress Score: Higher score is worse.
        # We want to penalize low/negative alpha and reward high beta.
        # Weights are subjective but can be tuned. Here, they are equal.
        timeline_df['distress_score'] = (timeline_df['beta'] * 0.5) - (timeline_df['alpha'] * 0.5)

        # Normalize the score to a 0-1 "probability" scale for easier interpretation
        min_score = timeline_df['distress_score'].min()
        max_score = timeline_df['distress_score'].max()
        timeline_df['failure_probability'] = (timeline_df['distress_score'] - min_score) / (max_score - min_score)

        # Find the date with the highest score
        peak_distress_date = timeline_df['failure_probability'].idxmax()

        return timeline_df, peak_distress_date.date()

        
    def load_and_merge_data(self, 
                            risk_free_rate_ticker: str = '^IRX') -> pd.DataFrame:
        """
        Loads all necessary data files, merges them, and returns a single DataFrame.
        
        Args:
            data_dir: The directory containing the CSV data files.

        Returns:
            A merged pandas DataFrame containing share prices, index values, and reference data.
        """
        logging.info("Starting data loading and merging process...")

        # Load Share Prices
        share_price_files = self._find_files('sharePricePart')
        df_price = pd.concat(
            [pd.read_csv(f, 
                        parse_dates=['pricing_date'], 
                        date_format=self.price_date_format,
                        ) for f in share_price_files]
        ).sort_values(by=['companyid', 'pricing_date'])
        logging.info(f"Loaded {len(df_price)} rows from {len(share_price_files)} share price files.")

        # Load Index Values
        index_files = self._find_files('indexValuesPart')
        df_index = pd.concat(
            [pd.read_csv(f, 
                        parse_dates=['pricing_date'], 
                        date_format=self.index_date_format,
                        ) for f in index_files]
        )
        # Rename for clarity in merge
        df_index = df_index.rename(columns={'indexid': 'index_id'})
        logging.info(f"Loaded {len(df_index)} rows from {len(index_files)} index value files.")



        # Load Reference Data to link company to index
        ref_file = self._find_files('referenceDataPart')[0]
        df_ref = pd.read_csv(ref_file)
                
        logging.info(f"Loaded reference data from {ref_file}.")

        # Merge data
        df = (df_price.merge(df_ref[['companyid', 'indexid']]
                            .rename(columns={'indexid': 'index_id'}), 
                            on='companyid', how='left')
            .merge(df_index, on=['index_id', 'pricing_date'], how='left')
            .dropna(subset=['index_value_norm', 'adjusted_price_local_norm'])
            .sort_values(by=['companyid', 'pricing_date'])
            .rename(columns={'stock': 'adjusted_price_local_norm',
                            'index': 'index_value_norm'})
            .set_index('pricing_date')
            )

        df = df[df.index >= pd.to_datetime('2010-01-01')]
        
        # Fetch risk-free rate (3-Month Treasury Bill)
        start_date = df.index.min()
        end_date = df.index.max()
        logging.info(f"Fetching risk-free rate from {start_date} to {end_date}...")
        risk_free_rate_data = yf.download(risk_free_rate_ticker, 
                                            start=start_date, 
                                            end=end_date, 
                                            progress=False)['Adj Close']        
        
        df = (df.join(risk_free_rate_data.rename('risk_free_rate'))
            .ffill()
            .dropna()
            .reset_index()
            )

        logging.info(f"Final merged DataFrame has {len(df)} rows after dropping NaNs in key columns.")

        # Get failure dates 
        logging.info("Identifying max 'beta' dates for failed companies...")
        failure_list_file = self._find_files('corporateFailureList')[0]
        df_failures = (pd.read_csv(failure_list_file)
                    .rename(columns={'peakyear': 'peakYear'})
                    .assign(peakYear=lambda x: pd.to_datetime(x['peakYear'], 
                                                                format='%Y'))
                    )
        
        failure_dates = {}
        distress_timeline = {}
        dfg = df.groupby('companyid')
        for companyid, group in dfg:
            try:
                failure_year = df_failures[df_failures['companyid'] == companyid]['peakYear'].values[0]
                if not failure_year:
                    continue
            except:
                continue    

            start_date = failure_year.year
            end_date = failure_year.year + 1

            df_group = group[(group['pricing_date'].dt.year >= start_date) & (group['pricing_date'].dt.year < end_date)]
            timeline_df, peak_distress_date = self.compute_failure_probability_timeline(df_group, failure_year)

            failure_dates[companyid] = failure_year
            distress_timeline[companyid] = timeline_df

        logging.info(f"Successfully computed failure dates and distress timelines!")
        
        return df, failure_dates, distress_timeline


    def load_signals_with_labels(self, failure_dates: dict) -> pd.DataFrame:
        """
        Adds the 'fail' label to the binary signals data based on the failure_dates.
        
        Args:
            data_dir: The directory containing the binarySignalsPart files.
            failure_dates: Dictionary from get_failure_dates.
            fail_days: The number of days after max_beta_date to label as failure.

        Returns:
            A DataFrame of binary signals with an added 'fail' column.
        """
        logging.info(f"Adding failure labels for a {self.fail_days}-day window...")
        signal_files = self._find_files('binarySignalsPart')
        if not signal_files:
            raise FileNotFoundError(f"No binarySignalsPart files found in {self.data_dir}")

        df_signals = pd.concat(
            [pd.read_csv(f, 
                        parse_dates=['signal_date'], 
                        date_format=self.signal_date_format) for f in signal_files]
        ).sort_values(by=['companyid', 'signal_date'])

        fail_labels = []
        time_delta = timedelta(days=self.fail_days)

        for _, row in df_signals.iterrows():
            is_failure = 0
            if row.companyid in failure_dates:
                max_date = failure_dates[row.companyid]
                # The failure window is AFTER the max_date
                if max_date < row.signal_date <= (max_date + time_delta):
                    is_failure = 1
            fail_labels.append(is_failure)
        
        df_signals['fail'] = fail_labels
        logging.info(f"Labeled {sum(fail_labels)} records as failure=1.")

        return df_signals


    # --- Failure Labeling (Current Flawed Logic) ---
    def get_max_beta_dates(self, merged_df: pd.DataFrame) -> dict:
        """
        Identifies the date of "maximum beta" for each failed company.
        
        NOTE: This function implements the flawed "beta" logic for faithful reproduction.
        
        Args:
            merged_df: The DataFrame from load_and_merge_data.
            corporate_failure_list_path: Path to the corporateFailureList CSV.

        Returns:
            A dictionary mapping companyid to its max "beta" date.
        """

        # Consider only failures post-2010 as requested
        failure_list_file = self._find_files('corporateFailureList')[0]
        df_failures = (pd.read_csv(failure_list_file)
                    .rename(columns={'peakyear': 'peakYear'})
                    .assign(peakYear=lambda x: pd.to_datetime(x['peakYear'], 
                                                                format=self.date_format))
                    )
        
        max_year_map = {row.companyid: row.peakYear for _, row in df_failures.iterrows()}
        
        df = merged_df[merged_df['companyid'].isin(max_year_map.keys())].copy()
        
        # Calculate the flawed "beta"
        df['pseudo_beta'] = df['adjusted_price_local_norm'] / df['index_value_norm'] - 1.0
        
        max_beta_dates = {}
        
        for company_id, group in df.groupby('companyid'):
            failure_year = max_year_map.get(company_id)
            if not failure_year:
                continue
                
            year_group = group[group['pricing_date'].dt.year == failure_year]
            if not year_group.empty:
                max_beta_row = year_group.loc[year_group['pseudo_beta'].idxmax()]
                max_beta_dates[company_id] = max_beta_row['pricing_date']
                
        logging.info(f"Found max 'beta' dates for {len(max_beta_dates)} companies.")
        return max_beta_dates


    def compute_weights(self, df: pd.DataFrame, 
                        target_col: str='fail') -> pd.Series:
        """
        Computes the signal weights based on the current model's logic.
        
        Args:
            df: The DataFrame with failure labels.

        Returns:
            A pandas Series containing the weight for each signal.
        """
        logging.info("Computing signal weights...")

        signal_cols, failed_group, non_failed_group = self._split_data(df, target_col)

        failed_props = failed_group[signal_cols].mean()
        non_failed_props = non_failed_group[signal_cols].mean()
        
        # IMPORTANT: Using the statistically incorrect std dev to match the original model
        non_failed_std = non_failed_group[signal_cols].std()
        
        # Avoid division by zero
        non_failed_std[non_failed_std == 0] = 1e-6
        
        weights = (failed_props - non_failed_props) / non_failed_std
        logging.info("Successfully computed weights for all signals.")

        self._save_artifacts(weights=weights)   

        return weights


    def calculate_z_scores(self, df: pd.DataFrame, target_col: str='fail') -> pd.Series:
        """
        Calculates a score for each feature based on the Z-score for the 
        difference in proportions between two groups defined by a target column.

        A higher score indicates a greater separation (sensitivity) between the groups.

        Args:
            df: Pandas DataFrame containing binary feature columns and a binary target column.
            target_col: The name of the binary target column (e.g., 'fail').

        Returns:
            A pandas DataFrame with feature names and their calculated scores,
            sorted in descending order.
        """

        # Separate into failure (group 1) and non-failure (group 0)
        feature_cols, group_1, group_0 = self._split_data(df, target_col)
        weights = {}

        

        n_1 = len(group_1)
        n_0 = len(group_0)

        # Handle edge case where one group is empty
        if n_1 == 0 or n_0 == 0:
            print("Warning: One of the target groups is empty. Cannot calculate scores.")
            return pd.Series(weights)

        for feature in feature_cols:
            # Calculate proportions of '1's in each group
            p_1 = group_1[feature].mean()
            p_0 = group_0[feature].mean()

            # Calculate pooled proportion
            p_pooled = ((p_1 * n_1) + (p_0 * n_0)) / (n_1 + n_0)

            # Calculate standard error of the difference
            se_diff = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_1 + 1/n_0))

            # Handle case where standard error is zero to avoid division by zero
            if se_diff == 0:
                score = 0.0
            else:
                # Calculate Z-score and take its absolute value for the final score
                z_score = (p_1 - p_0) / se_diff
                score = np.abs(z_score)

            weights[feature] = score

        # Create a DataFrame from the results and sort it
        weights_df = pd.Series(weights)
        weights_df = weights_df.sort_values(ascending=False)

        self._save_artifacts(weights=weights_df)

        return weights_df

    def add_score(self, df: pd.DataFrame, 
                  weights: pd.Series,
                  target_col: str='fail') -> pd.DataFrame:
        """
        Adds the failure score to the DataFrame.
        
        Args:
            df: The DataFrame with failure labels.
            weights: The computed signal weights.

        Returns:
            The DataFrame with an added 'score' column.
        """
        logging.info("Calculating failure scores for all records...")
        signal_cols = weights.index
        
        # Ensure columns are in the same order for dot product
        df_copy = df[['companyid', 'signal_date', target_col]].copy()
        df_copy['score'] = df[signal_cols].dot(weights)
        
        self._save_artifacts(scores=df_copy)

        logging.info("Finished calculating scores.")

        return df_copy

                
# --- Main Execution ---
if __name__ == '__main__':
    pass
