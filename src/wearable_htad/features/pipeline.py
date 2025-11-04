from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wearable_htad.data.preprocessing import load_accelerometer_txt

# from wearable_htad.data.windowing import segment_windows
from wearable_htad.features.extract_accel_features import extract_features_from_window


def _infer_fs_from_df(df: pd.DataFrame, default_fs: float = 28.5) -> float:
    """Infer sampling rate (Hz) from a 'timestamp' column if present (ms).

    Falls back to default_fs when missing or degenerate.
    """
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
        dt = np.diff(ts[np.isfinite(ts)])
        if dt.size and np.median(dt) > 0:
            return float(1000.0 / np.median(dt))  # ms -> Hz
    return float(default_fs)


def segment_windows(
    df: pd.DataFrame, window_size: int, overlap: float
) -> list[tuple[int, int, pd.DataFrame]]:
    """Segment into overlapping windows.

    Returns a list of (start_idx, end_idx, window_df).
    - window_size: number of samples per window (NOT seconds).
    - overlap: fraction in [0, 1); e.g., 0.5 means 50% overlap.
    """
    assert window_size > 1, "window_size must be > 1 (samples)"
    assert 0.0 <= overlap < 1.0, "overlap must be in [0, 1)"
    step = max(1, int(window_size * (1.0 - overlap)))

    n = len(df)
    out: list[tuple[int, int, pd.DataFrame]] = []
    i = 0
    while i + window_size <= n:
        j = i + window_size
        out.append((i, j, df.iloc[i:j]))
        i += step
    return out


def extract_features_from_segmented_signal(
    df: pd.DataFrame,
    window_size: int,
    overlap: float,
    fs: float | None = None,
) -> pd.DataFrame:
    """Segments df into overlapping windows and extracts features for each window.

    Adds per-window metadata: row_id, start_idx, start_time (if timestamp present).
    """
    # Determine sampling rate
    fs_val = float(fs) if fs is not None else _infer_fs_from_df(df)

    windows = segment_windows(df, window_size=window_size, overlap=overlap)

    rows = []
    for k, (i, _j, w) in enumerate(windows):
        feats = extract_features_from_window(w, fs=fs_val)  # <-- your upgraded function
        feats["row_id"] = k
        feats["start_idx"] = i
        if "timestamp" in w.columns:
            # store first timestamp of the window for later alignment with audio
            feats["start_time"] = pd.to_numeric(w["timestamp"].iloc[0], errors="coerce")
        rows.append(feats)

    return pd.DataFrame(rows)


def process_accelerometer_file(
    path: str | Path,
    window_size: int,
    overlap: float,
    fs: float | None = None,
) -> pd.DataFrame:
    """Load an accelerometer file and returns a DataFrame of features per window.

    Includes activity, sensor, timestamp (first value) as metadata on each row.
    """
    df = load_accelerometer_txt(
        path
    )  # your existing loader: must produce x,y,z,(timestamp, activity, sensor)

    features = extract_features_from_segmented_signal(
        df,
        window_size=window_size,
        overlap=overlap,
        fs=fs,  # will be inferred from 'timestamp' if None
    )

    # Broadcast simple file-level metadata (constant across windows)
    for key in ["activity", "sensor"]:
        if key in df.columns:
            features[key] = df[key].iloc[0]
    if "timestamp" in df.columns and "start_time" not in features.columns:
        features["start_time"] = pd.to_numeric(df["timestamp"].iloc[0], errors="coerce")

    # Keep a reference to file path (useful for debugging)
    features["source_file"] = str(path)

    return features


def build_full_feature_dataset(
    folder: str | Path,
    window_size: int,
    overlap: float,
    fs: float | None = None,  # allow override; otherwise infer per file
) -> pd.DataFrame:
    """Process all raw accelerometer files and return a full feature dataset.

    Iterates user subfolders and aggregates features, adding 'user'.
    """
    folder = Path(folder)
    all_features: list[pd.DataFrame] = []

    for user_folder in folder.iterdir():
        if not user_folder.is_dir():
            continue
        user_id = user_folder.name

        for path in user_folder.glob("*.txt"):
            try:
                feats = process_accelerometer_file(
                    path=path,
                    window_size=window_size,
                    overlap=overlap,
                    fs=fs,
                )
                feats["user"] = user_id
                all_features.append(feats)
            except Exception as e:
                print(f"Skipping {path}: {e}")

    if not all_features:
        # Avoid concat([]) error; return empty frame with expected columns
        return pd.DataFrame()

    df_all = pd.concat(all_features, ignore_index=True)

    # Final hygiene: drop constant columns, keep NaN fill minimal (let model choose)
    non_meta = [
        c
        for c in df_all.columns
        if c
        not in {"user", "activity", "sensor", "source_file", "row_id", "start_idx", "start_time"}
    ]
    # drop near-constant features to reduce noise
    keep = [
        c for c in non_meta if c in df_all and np.nanstd(df_all[c].to_numpy(dtype=float)) > 1e-8
    ]
    df_all = pd.concat([df_all[keep], df_all.drop(columns=keep, errors="ignore")], axis=1)

    return df_all
