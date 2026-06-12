
import os
import pandas as pd
import numpy as np
import pickle

from scipy.stats import ks_2samp

# top features from notebook's feature importance charts
TOP_FEATURES = ['Amount', 'V17', 'V12', 'V14', 'V10', 'V11', 'V16', 'V3']

def load_reference_data():
    # drift_detection.py is at ROOT
    # models/ is also at ROOT
    # So just go to same level!
    
    base_dir = os.path.dirname(os.path.abspath(__file__))  # ROOT folder
    path     = os.path.join(base_dir, 'models', 'reference_data.csv')
    
    print(f"Loading reference data from: {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found at: {path}")
    
    return pd.read_csv(path)


def detect_drift(current_df: pd.DataFrame) -> dict:
    """
    Takes new incoming data and checks if it has drifted
    from the original training distribution.

    Parameters:
        current_df : new data (uploaded CSV or API payload)

    Returns:
        A dictionary with drift results per feature + overall status
    """

    # Load the saved reference data from disk
    reference_df = load_reference_data()

    results = []         # will store result for each feature
    any_drift = False    # tracks if at least one feature drifted

    for feature in TOP_FEATURES:
        # KS test = Kolmogorov-Smirnov test
        # It compares TWO distributions and tells you if they are significantly different
        # stat  = how different they are (higher = more different)
        # pvalue = probability they come from same distribution
        #          if pvalue < 0.05 → distributions are DIFFERENT → DRIFT
        stat, pvalue = ks_2samp(
            reference_df[feature].dropna(),
            current_df[feature].dropna()
        )

        drifted = pvalue < 0.05   # True if drift detected in this feature

        if drifted:
            any_drift = True      # at least one feature drifted → overall drift

        results.append({
            'feature': feature,
            'ks_stat': round(float(stat), 4),    # how big is the drift
            'p_value': round(float(pvalue), 4),  # statistical confidence
            'drifted': drifted                   # True/False
        })

    return {
        'drift_detected': any_drift,   # overall verdict
        'feature_results': results,    # per-feature breakdown
        'drifted_features': [          # list of just the drifted ones
            r['feature'] for r in results if r['drifted']
        ]
    }