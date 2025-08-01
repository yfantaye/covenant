import logging
from .updated_failure import ScottStrategy

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

MODEL_MAPPING = {
    "scottv1": ScottStrategy,
    "scottv2": ScottStrategy
}

def run(config):
    '''
    Runs the model pipeline.
    '''
    model_class = MODEL_MAPPING[config['model_type']]
    sm = model_class(config)

    df_merged, failure_dates, distress_timeline = sm.load_and_merge_data()

    if config['model_type'] in ['scottv1']:
        failure_dates = sm.get_max_beta_dates(df_merged)

    # Reads signal data from file and add failure labels
    df_labeled = sm.load_signals_with_labels(failure_dates)

    if config['model_type'] in ['scottv1']:
        weights = sm.compute_weights(df_labeled)
    else:
        weights = sm.calculate_z_scores(df_labeled)
    
    df_labeled = sm.add_score(df_labeled, weights)

    return df_labeled