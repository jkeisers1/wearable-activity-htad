import pandas as pd
import numpy as np
from scipy.stats import entropy
from typing import Union


def extract_features_from_window(window: pd.DataFrame) -> pd.Series:
    """
    Extract time-domain features from a single accelerometer window.

    Args:
        window: A DataFrame containing columns ['x', 'y', 'z']

    Returns:
        A Series of extracted features.
    """
    feats = {}

    for axis in ['x', 'y', 'z']:
        signal = window[axis].values

        feats[f'{axis}_mean'] = np.mean(signal)
        feats[f'{axis}_std'] = np.std(signal)
        feats[f'{axis}_min'] = np.min(signal)
        feats[f'{axis}_max'] = np.max(signal)

        feats[f'{axis}_energy'] = np.sum(signal ** 2) / len(signal)

        # Bin the signal and compute Shannon entropy
        hist, _ = np.histogram(signal, bins=10, density=True)
        feats[f'{axis}_entropy'] = entropy(hist + 1e-6)  # avoid log(0)

    # Signal Magnitude Area
    sma = np.mean(np.abs(window["x"]) + np.abs(window["y"]) + np.abs(window["z"]))
    feats["sma"] = sma

    return pd.Series(feats)
