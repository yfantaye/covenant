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
        self.global_output_dir = config.get('global_output_directory', './output')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.global_output_dir, exist_ok=True)
        self.use_sample_data = config.get('use_sample_data', False)
        self.date_range_start = config.get('date_range_start', '2010-01-01')
        self.date_range_end = config.get('date_range_end', '2024-12-31')
        self.default_risk_free_rate = config.get('default_risk_free_rate', 1.01)
        # --- Model Specific Parameters ---
        
        self.date_format = config.get('modelv1').get('date_format', '%m/%d/%Y')
        self.index_date_format = config.get('index_date_format', '%Y-%m-%d')
        self.price_date_format = config.get('price_date_format', '%m/%d/%Y')
        self.signal_date_format = config.get('signal_date_format', '%m/%d/%Y')
        
        self.colmap = config.get('modelv1').get('colmap', {})
        self.fail_days = config.get('modelv1').get('fail_days', 180)
        self.trading_days = config.get('modelv1').get('trading_days', 252)
        self.rolling_window_days = config.get('modelv1').get('rolling_window_days', 90)
        self.normalize_weights = config.get('modelv1').get('normalize_weights', True)



    def _save_artifacts(self, 
                        scores: pd.DataFrame = None, 
                        weights: pd.Series = None
                        ):
        """Saves the trained model and evaluation results."""
        if scores is not None:          
            scores_rounded = scores.copy()
            if 'score' in scores_rounded.columns:
                scores_rounded['score'] = scores_rounded['score'].round(4)
            scores_rounded.to_csv(os.path.join(self.output_dir, "scores.csv"), index=False)
        if weights is not None:
            try:
                weights.index.name = 'Feature'
            except:
                pass

            cmap = {0: 'Weight', 
                    weights.name or 0: 'Weight'}
            
            print(f'Saving weights to {os.path.join(self.output_dir, "weights.csv")}')
            (weights
             .round(4)
             .reset_index()
             .rename(columns=cmap)
             .to_csv(os.path.join(self.output_dir, "weights.csv"), index=False)
             )

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

    def _get_risk_free_rate(self, 
                            start_date: str, 
                            end_date: str,
                            risk_free_rate_ticker: str = '^IRX') -> pd.DataFrame:
        """
        Fetches risk-free rate data from Yahoo Finance.
        """
        logging.info(f"Fetching risk-free rate from {start_date} to {end_date}...")

        # Cache risk-free rate data to file and load if exists and not empty
        cache_dir = os.path.join(self.global_output_dir, "risk_free_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_filename = f"risk_free_{risk_free_rate_ticker}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        cache_path = os.path.join(cache_dir, cache_filename)

        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            logging.info(f"Loading risk-free rate data from cache: {cache_path}")
            risk_free_rate_data = (pd.read_csv(cache_path,
                                               parse_dates=['Date'])
                                   ).set_index('Date')

        else:
            try:
                risk_free_rate_data = yf.download(
                    risk_free_rate_ticker,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False
                )
                if not risk_free_rate_data.empty:
                    if isinstance(risk_free_rate_data.columns, pd.MultiIndex):
                        risk_free_rate_data.columns = risk_free_rate_data.columns.get_level_values(0)
                    risk_free_rate_data.to_csv(cache_path)
                    logging.info(f"Fetched {len(risk_free_rate_data)} rows of risk-free rate data.")
                else:
                    logging.warning(f"Downloaded risk-free rate data is empty for {risk_free_rate_ticker} from {start_date} to {end_date}.")
                    risk_free_rate_data = None
            except Exception as e:
                logging.info("Error fetching data for symbol={risk_free_rate_ticker}: {e}")
                risk_free_rate_data = None                

        return risk_free_rate_data
    
    def compute_failure_probability_timeline(self, 
                                             dfin: pd.DataFrame, 
                                             failure_year: int, 
                                             companyid: str="") -> tuple:
        """
        Calculates a daily "Financial Distress Score" based on rolling alpha and beta
        to identify the most probable time that failure became critical.

        Args:
            df: The DataFrame with the stock, index, and risk-free rate data.
            failure_year: The year the company failed.

        Returns:
            tuple: A pandas DataFrame with the daily timeline and the date of the highest distress score.
        """ 

        df = dfin.copy()          
        
        # Combine, align, and calculate returns
        df['stock_return'] = np.log(df['stock'] / df['stock'].shift(1))
        df['index_return'] = np.log(df['index'] / df['index'].shift(1))
        df = df.dropna()  

        # Calculate rolling alpha and beta (90-day window is common)
        window = self.rolling_window_days
        rolling_results = []

        #
        dfyear = df['pricing_date'].dt.year

        for i in range(len(df) - window):
            subset = df.iloc[i:i+window]
            date_window = df['pricing_date'].iloc[i + window]
            year_window = dfyear.iloc[i + window]

            # Filter for the actual failure year
            if year_window != failure_year:
                continue
            
            # Calculate beta for the window
            slope, _, _, _, _ = stats.linregress(x=subset['index_return'], 
                                                y=subset['stock_return'])
            beta = slope
            
            # Calculate alpha for the window (annualized)
            trading_days = self.trading_days
            annual_stock_return = subset['stock_return'].mean() * trading_days
            annual_index_return = subset['index_return'].mean() * trading_days
            if 'risk_free_rate' in subset.columns:
                annual_risk_free_rate = subset['risk_free_rate'].mean() / 100
            else:
                annual_risk_free_rate = self.default_risk_free_rate

            expected_return = annual_risk_free_rate + beta * (annual_index_return - annual_risk_free_rate)
            alpha = annual_stock_return - expected_return
            
            rolling_results.append({
                'date': date_window,
                'alpha': alpha,
                'beta': beta
            })

        if not rolling_results:
            print("Could not generate rolling results. Check data availability.")
            return None, None
            
        timeline_df = pd.DataFrame(rolling_results).set_index('date')
            

        if timeline_df.empty:
            print(timeline_df.info())
            print(f'No data for companyid={companyid} in year={failure_year.year}')
            print(f'Timeline DF Unique Years: {pd.Timestamp(timeline_df.index).year.unique()}')
            return None, None
        
        # Calculate Financial Distress Score: Higher score is worse.
        # We want to penalize low/negative alpha and reward high beta.
        # Weights are subjective but can be tuned. Here, they are equal.
        timeline_df['distress_score'] = (timeline_df['beta'] * 0.5) - (timeline_df['alpha'] * 0.5)

        # Normalize the score to a 0-1 "probability" scale for easier interpretation
        min_score = timeline_df['distress_score'].min()
        max_score = timeline_df['distress_score'].max()
        timeline_df['failure_probability'] = (timeline_df['distress_score'] - min_score) / (max_score - min_score)

        # Find the date with the highest score
        if timeline_df.empty:
            peak_distress_date = None
        else:
            peak_distress_date = timeline_df['failure_probability'].idxmax().date()

        return peak_distress_date, timeline_df


    
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
        share_price_files = self._find_files('sharePrice')
        df_price = pd.concat(
            [pd.read_csv(f, 
                        parse_dates=['pricing_date'], 
                        date_format=self.price_date_format,
                        ) for f in share_price_files]
        ).sort_values(by=['companyid', 'pricing_date'])
        logging.info(f"Loaded {len(df_price)} rows from {len(share_price_files)} share price files.")

        # Load Index Values
        index_files = self._find_files('indexValues')
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
        ref_file = self._find_files('referenceData')[0]
        df_ref = pd.read_csv(ref_file)        

        logging.info(f"Loaded reference data from {ref_file}.")


        # Merge data
        df = (df_price.merge(df_ref[['companyid', 'indexid']]
                            .rename(columns={'indexid': 'index_id'}), 
                            on='companyid', how='left')
            .merge(df_index, on=['index_id', 'pricing_date'], how='left')
            .dropna(subset=['index_value_norm', 'adjusted_price_local_norm'])
            .sort_values(by=['companyid', 'pricing_date'])
            .rename(columns={'adjusted_price_local_norm': 'stock',
                             'adjusted_price_usd_norm': 'stock_usd',
                            'index_value_norm': 'index'})
            .set_index('pricing_date')
            )

        # Filter by date range
        df = df[df.index >= pd.to_datetime(self.date_range_start)]
        df = df[df.index <= pd.to_datetime(self.date_range_end)]
        
        # Fetch risk-free rate (3-Month Treasury Bill)
        start_date = df.index.min()
        end_date = df.index.max()

        risk_free_rate_data = self._get_risk_free_rate(start_date, end_date, risk_free_rate_ticker)
        if risk_free_rate_data is not None:
            risk_free_rate_data = risk_free_rate_data['Close']   

            df = (df.join(risk_free_rate_data.rename('risk_free_rate'))
                    .ffill()
                    .dropna()
                 )                     
        else:
            logging.warning(f"No risk-free rate data found for {risk_free_rate_ticker} from {start_date} to {end_date}.")

        logging.info(f"Final merged DataFrame has {len(df)} rows after dropping NaNs in key columns.")

        df = df.reset_index(drop=False)

        return df


    def load_signals_with_labels(self, failure_dates: dict) -> pd.DataFrame:
        """
        Adds the 'fail' label to the binary signals data based on the failure_dates.
        
        Args:
            data_dir: The directory containing the binarySignals files.
            failure_dates: Dictionary from get_failure_dates.
            fail_days: The number of days after max_beta_date to label as failure.

        Returns:
            A DataFrame of binary signals with an added 'fail' column.
        """
        logging.info(f"Adding failure labels for a {self.fail_days}-day window...")
        signal_files = self._find_files('binarySignals')
        if not signal_files:
            raise FileNotFoundError(f"No binarySignals files found in {self.data_dir}")

        df_signals = pd.concat(
            [pd.read_csv(f, 
                        parse_dates=['signal_date'], 
                        date_format=self.signal_date_format) for f in signal_files]
        ).sort_values(by=['companyid', 'signal_date'])

        # Filter by date range
        df_signals = df_signals[df_signals['signal_date'] >= pd.to_datetime(self.date_range_start)]
        df_signals = df_signals[df_signals['signal_date'] <= pd.to_datetime(self.date_range_end)]


        time_delta = timedelta(days=self.fail_days)


        # Build arrays for max_date and window_end for each row
        
        df_list = []
        time_delta = timedelta(days=self.fail_days)
        for cid, dfgroup in df_signals.groupby('companyid'):
            group = dfgroup.copy()
            max_date= failure_dates.get(cid, None)
            if max_date:
                c1 = max_date < group['signal_date']
                c2 = group['signal_date'] <= max_date + time_delta
                group['fail'] = c1 & c2
                df_list.append(group[c2])                
            else:
                group['fail'] = 0
                df_list.append(group)   
                
        df_signals = pd.concat(df_list, axis=0)


        logging.info(f"Labeled {sum(df_signals['fail'])} records as failure=1.")

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
                                                                format="%Y"))
                    )
        
        # Filter by date range
        df_failures = df_failures[df_failures['peakYear'] >= pd.to_datetime(self.date_range_start)]
        df_failures = df_failures[df_failures['peakYear'] <= pd.to_datetime(self.date_range_end)]
        
        max_year_map = {row.companyid: row.peakYear for _, row in df_failures.iterrows()}
        
        df = merged_df[merged_df['companyid'].isin(max_year_map.keys())].copy()
        
        max_beta_dates = {}
        
        for company_id, df_group in df.groupby('companyid'):

            group = df_group.copy()

            failure_year = max_year_map.get(company_id)
            if not failure_year:                
                continue
        

            # Compute pseudo_beta robustly: avoid division by zero or NaN
            index_safe = group['index'].replace(0, np.nan)
            pseudo_beta = (group['stock'] / index_safe) - 1.0
            pseudo_beta = pseudo_beta.replace([np.inf, -np.inf], np.nan).fillna(0)
            group['pseudo_beta'] = pseudo_beta
                
 
            year_group = group[group['pricing_date'].dt.year == failure_year.year]
            if not year_group.empty:
                max_beta_row = year_group.loc[year_group['pseudo_beta'].idxmax()]
                max_beta_dates[company_id] = max_beta_row['pricing_date']
            else:
                logging.info(f"No data for companyid={company_id} in year={failure_year}")
                print('Merged DF Unique years:')
                print(group['pricing_date'].dt.year.unique())
                
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

        if self.normalize_weights:
            # Normalize so all values are between 0 and 1 and sum to 1
            if weights.max() > 0:
                weights = (weights - weights.min()) / (weights.max() - weights.min())
            weights_sum = weights.sum()
            if weights_sum > 0:
                weights = weights / weights_sum


        logging.info("Successfully computed weights for all signals.")

        self._save_artifacts(weights=weights)   

        return weights


    def compute_probable_failure_dates(self, df: pd.DataFrame) -> dict:
        """
        Computes the failure dates for each company.
        """

        # Get failure dates 
        logging.info("Identifying max 'beta' dates for failed companies...")
        failure_list_file = self._find_files('corporateFailureList')[0]
        df_failures = pd.read_csv(failure_list_file).rename(columns={'peakyear': 'peakYear'})
        # Ensure 'peakYear' is datetime (year only)
        if not np.issubdtype(df_failures['peakYear'].dtype, np.datetime64):
            # Try to convert to int first, then to datetime
            df_failures['peakYear'] = pd.to_datetime(df_failures['peakYear'].astype(str), format='%Y', errors='coerce')
        

        failure_dates = {}
        distress_timeline = {}


        dfg = df.groupby('companyid')
        for companyid, group in dfg:
            try:
                failure_year = df_failures[df_failures['companyid'] == companyid]['peakYear'].values[0]
                if isinstance(failure_year, np.datetime64):
                    failure_year = pd.to_datetime(failure_year)
                if not failure_year:
                    continue
            except:
                continue    

            start_date = failure_year.year
            end_date = failure_year.year + 1

            c1 = (group['pricing_date'].dt.year >= start_date)
            c2 = (group['pricing_date'].dt.year < end_date)
            df_group = group[c1 & c2]

                    
            peak_distress_date, timeline_df = self.compute_failure_probability_timeline(group, 
                                                                                        failure_year.year, 
                                                                                        companyid)

            if peak_distress_date is not None:
                failure_dates[companyid] = failure_year
            if timeline_df is not None:
                distress_timeline[companyid] = timeline_df

        logging.info(f"Successfully computed failure dates and distress timelines!")

        return failure_dates, distress_timeline


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
            p_pooled = np.clip(p_pooled, 1e-10, 1.0 - 1e-10)  # Avoid exact 0 or 1

            # Calculate standard error with error handling
            try:
                se_diff = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_1 + 1/n_0))
                se_diff = np.nan_to_num(se_diff, nan=0.0, posinf=0.0, neginf=0.0)
            except:
                se_diff = 0.0

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
        if self.normalize_weights:
            # Normalize so all values are between 0 and 1 and sum to 1
            if weights_df.max() > 0:
                weights_df = (weights_df - weights_df.min()) / (weights_df.max() - weights_df.min())
            weights_sum = weights_df.sum()
            if weights_sum > 0:
                weights_df = weights_df / weights_sum

        weights_df = weights_df.sort_values(ascending=False)

        print(f'Weights DF: {weights_df}')
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
        df_copy['score'] = df[signal_cols].dot(weights).round(4)

        print(f'DF Copy: ')
        print(df_copy.info())
        
        self._save_artifacts(scores=df_copy)

        logging.info("Finished calculating scores.")

        return df_copy

                
# --- Main Execution ---
if __name__ == '__main__':
    pass
