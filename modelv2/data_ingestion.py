import pandas as pd
import glob
import os
import logging

def _load_and_concat(data_dir: str, pattern: str, sample: bool = False) -> pd.DataFrame:
    """Helper to load and concatenate CSVs matching a pattern."""
    files = glob.glob(os.path.join(data_dir, f'{pattern}*.csv'))
    if not files:
        logging.warning(f"No files found for pattern: {pattern}")
        return pd.DataFrame()
    
    df = pd.concat(
        [pd.read_csv(f, low_memory=False) for f in files],
        ignore_index=True
    )

    # Add sample specific date format and column mapping
    date_format = '%Y-%m-%d'
    colmap = {}
    if sample:
        date_format = '%m/%d/%Y'
        colmap = {"indexvalues": "indexValues"}
        
    # Convert date columns if they exist
    for col in df.columns:
        if 'date' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce', format=date_format)
                if colmap:
                    df = df.rename(columns=colmap)
            except Exception:
                pass # Ignore columns that can't be converted
    return df

def load_all_data(data_dir: str, **kwargs) -> dict:
    """
    Loads all data sources from the specified directory into a dictionary of DataFrames.
    """
    data_dict = {
        "binary_signals": _load_and_concat(data_dir, "binarySignalsPart", **kwargs),
        "raw_subsignals": _load_and_concat(data_dir, "rawSubsignalsPart", **kwargs),
        "share_price": _load_and_concat(data_dir, "sharePricePart", **kwargs),
        "index_values": _load_and_concat(data_dir, "indexValuesPart", **kwargs),
        "reference_data": _load_and_concat(data_dir, "referenceDataPart", **kwargs),
        "failure_list": _load_and_concat(data_dir, "corporateFailureList", **kwargs),
    }
    return data_dict
