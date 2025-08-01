import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats



def calculate_z_scores(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
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
    feature_cols = df.columns.drop(target_col)
    scores = {}

    # Separate into failure (group 1) and non-failure (group 0)
    group_1 = df[df[target_col] == 1]
    group_0 = df[df[target_col] == 0]

    n_1 = len(group_1)
    n_0 = len(group_0)

    # Handle edge case where one group is empty
    if n_1 == 0 or n_0 == 0:
        print("Warning: One of the target groups is empty. Cannot calculate scores.")
        return pd.DataFrame(list(scores.items()), columns=['Feature', 'Score'])

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

        scores[feature] = score

    # Create a DataFrame from the results and sort it
    scores_df = pd.DataFrame(list(scores.items()), columns=['Feature', 'Score'])
    scores_df = scores_df.sort_values(by='Score', ascending=False).reset_index(drop=True)

    return scores_df


def compute_alpha_beta(stock_data: pd.DataFrame, 
                       index_data: pd.DataFrame,
                       risk_free_rate_ticker: str = '^IRX'):
    """
    Computes the Alpha and Beta of a stock relative to a market index for a given period.

    Args:
        stock_ticker (str): The ticker symbol for the stock (e.g., 'META').
        index_ticker (str): The ticker symbol for the market index (e.g., 'SPY').
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        tuple: A tuple containing the calculated alpha and beta, or (None, None) if data is insufficient.
    """
    # 1. Fetch data from Yahoo Finance
    try:
        stock_data = stock_data.dropna()
        index_data = index_data.dropna()

        start_date = stock_data.index[0].strftime('%Y-%m-%d')
        end_date = stock_data.index[-1].strftime('%Y-%m-%d')
        
        # Fetch risk-free rate (3-Month Treasury Bill)
        risk_free_rate_data = yf.download(risk_free_rate_ticker, 
                                          start=start_date, 
                                          end=end_date, 
                                          progress=False)['Adj Close']
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None

    if stock_data.empty or index_data.empty:
        print("Could not download stock or index data for the given period.")
        return None, None

    # 2. Prepare and align data
    df = pd.DataFrame({'stock': stock_data, 'index': index_data}).dropna()
    df = df.join(risk_free_rate_data.rename('risk_free_rate')).ffill().dropna()

    # 3. Calculate daily returns
    df['stock_return'] = np.log(df['stock'] / df['stock'].shift(1))
    df['index_return'] = np.log(df['index'] / df['index'].shift(1))
    df = df.dropna()

    if len(df) < 2:
        print("Not enough overlapping data to perform calculation.")
        return None, None

    # 4. Calculate Beta (β)
    # Beta is the slope of the regression line between stock returns and index returns
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x=df['index_return'],
        y=df['stock_return']
    )
    beta = slope

    # 5. Calculate Alpha (α)
    # Annualize returns and risk-free rate for CAPM formula
    trading_days = 252
    annual_stock_return = df['stock_return'].mean() * trading_days
    annual_index_return = df['index_return'].mean() * trading_days
    # The risk-free rate from yfinance is already an annualized percentage
    annual_risk_free_rate = df['risk_free_rate'].mean() / 100

    # Alpha = Actual_Return - Expected_Return
    # Expected_Return = RiskFree_Rate + Beta * (Market_Return - RiskFree_Rate)
    expected_return = annual_risk_free_rate + beta * (annual_index_return - annual_risk_free_rate)
    alpha = annual_stock_return - expected_return

    return alpha, beta

def compute_failure_probability_timeline(stock_data: pd.DataFrame, 
                                         index_data: pd.DataFrame,
                                         failure_year: int,
                                         risk_free_rate_ticker: str = '^IRX'):
    """
    Calculates a daily "Financial Distress Score" based on rolling alpha and beta
    to identify the most probable time that failure became critical.

    Args:
        stock_ticker (str): The ticker symbol of the failed company.
        index_ticker (str): The ticker symbol for the market index.
        failure_year (int): The year the company failed.

    Returns:
        tuple: A pandas DataFrame with the daily timeline and the date of the highest distress score.
    """
    # Analyze the failure year and the year prior for a stable rolling window
    start_date = f"{failure_year - 1}-01-01"
    end_date = f"{failure_year}-12-31"


    # Fetch risk-free rate (3-Month Treasury Bill)
    risk_free_rate_data = yf.download(risk_free_rate_ticker, 
                                        start=start_date, 
                                        end=end_date, 
                                        progress=False)['Adj Close']    
    # Combine, align, and calculate returns
    df = pd.DataFrame({'stock': stock_data, 'index': index_data})
    df = df.join(risk_free_rate_data.rename('risk_free_rate')).ffill().dropna()
    df['stock_return'] = np.log(df['stock'] / df['stock'].shift(1))
    df['index_return'] = np.log(df['index'] / df['index'].shift(1))
    df = df.dropna()  

    # Calculate rolling alpha and beta (90-day window is common)
    window = 90
    rolling_results = []

    for i in range(len(df) - window):
        subset = df.iloc[i:i+window]
        
        # Calculate beta for the window
        slope, _, _, _, _ = stats.linregress(x=subset['index_return'], 
                                             y=subset['stock_return'])
        beta = slope
        
        # Calculate alpha for the window (annualized)
        trading_days = 252
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

if __name__ == "__main__":
    #---------------------------------------------------------------------
    # --- Example 0: Calculate feature scores for a sample DataFrame ---
    #---------------------------------------------------------------------

    # 1.1 Create a sample DataFrame
    np.random.seed(42)  # for reproducibility
    num_samples = 500
    num_features = 25

    data = {f'var_{i}': np.random.randint(0, 2, num_samples) for i in range(1, num_features + 1)}
    data['fail'] = np.random.randint(0, 2, num_samples)
    sample_df = pd.DataFrame(data)

    # To make it more interesting, let's make 'var_5' and 'var_18' more sensitive
    # Make var_5 more likely to be 1 when 'fail' is 1
    sample_df.loc[sample_df['fail'] == 1, 'var_5'] = np.random.choice([0, 1], size=len(sample_df[sample_df['fail'] == 1]), p=[0.2, 0.8])
    # Make var_18 more likely to be 0 when 'fail' is 1
    sample_df.loc[sample_df['fail'] == 1, 'var_18'] = np.random.choice([0, 1], size=len(sample_df[sample_df['fail'] == 1]), p=[0.7, 0.3])


    # 1.2. Calculate the scores
    feature_scores = calculate_z_scores(sample_df, 'fail')


    # 3. Print the results
    print("Feature Scores (Most Sensitive First):")
    print(feature_scores)

    #---------------------------------------------------------------------
    # --- Example 2: Calculate overall Alpha and Beta for META in 2022 ---
    #---------------------------------------------------------------------

    stock_ticker = 'META'
    index_ticker = 'SPY' # S&P 500 ETF as market proxy
    start_date = '2022-01-01'
    end_date = '2022-12-31'

    stock_data = yf.download(stock_ticker, 
                                start=start_date, 
                                end=end_date, 
                                progress=False)['Adj Close']
    index_data = yf.download(index_ticker, 
                                start=start_date, 
                                end=end_date, 
                                progress=False)['Adj Close']    

    alpha, beta = compute_alpha_beta(stock_data, index_data)

    if alpha is not None:
        print(f"--- Analysis for {stock_ticker} in 2022 ---")
        print(f"Beta (β): {beta:.4f}")
        print(f"Alpha (α): {alpha:.4f} or {alpha*100:.2f}%")
        print("\nInterpretation: In 2022, META was more volatile than the market (Beta > 1) and significantly underperformed its expected return (large negative Alpha).")


    # --- Example 2: Find the peak distress time for META in 2022 ---
    print("\n" + "="*50 + "\n")
    failure_year = 2022
    timeline, peak_date = compute_failure_probability_timeline(stock_ticker, index_ticker, failure_year)

    if timeline is not None:
        print(f"--- Financial Distress Timeline for {stock_ticker} in {failure_year} ---")
        print(f"Date of Peak Distress (Highest Probability Score): {peak_date}")
        
        # Show the data around the peak distress date
        print("\nData around the peak distress period:")
        print(timeline.loc[peak_date - pd.Timedelta(days=5):peak_date + pd.Timedelta(days=5)])    