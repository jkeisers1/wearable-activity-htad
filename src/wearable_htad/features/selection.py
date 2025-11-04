from __future__ import annotations

import numpy as np
import pandas as pd

AUDIO_PREFIX = "v1_"  # your MFCC columns
ACC_PREFIXES = ("x_", "y_", "z_", "mag_", "sma", "corr_")
META = {"row_id", "start_idx", "start_time", "sensor", "source_file", "activity", "user"}

def select_features(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Return numeric-only feature matrix for a given modality.

      - "accel": engineered accel features (x_, y_, z_, mag_, sma, corr_)
      - "audio": MFCC features (v1_*)
      - "fusion": both
    Drops metadata and replaces infs/NAs safely.
    """
    cols: list[str] = []
    if modality in ("accel", "fusion"):
        cols += [c for c in df.columns if c.startswith(ACC_PREFIXES)]
    if modality in ("audio", "fusion"):
        cols += [c for c in df.columns if c.startswith(AUDIO_PREFIX)]
    # keep only numeric + not meta
    cols = [c for c in cols if c not in META and np.issubdtype(df[c].dtype, np.number)]
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if X.shape[1] == 0:
        raise ValueError(f"No numeric features for modality='{modality}'. Check column names.")
    return X
