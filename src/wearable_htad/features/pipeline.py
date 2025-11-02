import pandas as pd

from typing import Union
from pathlib import Path
from wearable_htad.data.preprocessing import load_accelerometer_txt
from wearable_htad.data.windowing import segment_windows
from wearable_htad.features.extract import extract_features_from_window

def extract_features_from_segmented_signal(
    df: pd.DataFrame,
    window_size: int,
    overlap: float
) -> pd.DataFrame:
    """
    Segments the DataFrame into overlapping windows and extracts features from each window.

    Returns:
        A DataFrame of features, one row per window.
    """
    windows = segment_windows(df, window_size=window_size, overlap=overlap)
    feature_rows = [extract_features_from_window(w) for w in windows]
    return pd.DataFrame(feature_rows)

def process_accelerometer_file(
    path: Union[str, Path],
    window_size: int,
    overlap: float
) -> pd.DataFrame:
    """
    Loads an accelerometer file and returns a DataFrame of features per window.

    Includes activity, sensor, timestamp metadata.

    Returns:
        A DataFrame with features and labels.
    """
    df = load_accelerometer_txt(path)
    features = extract_features_from_segmented_signal(df, window_size, overlap)

    # Add metadata columns to each feature row
    for key in ["activity", "sensor", "timestamp"]:
        if key in df.columns:
            features[key] = df[key].iloc[0]

    return features

def build_full_feature_dataset(
    folder: Union[str, Path],
    window_size: int,
    overlap: float
) -> pd.DataFrame:
    """
    Process all raw accelerometer files and return a full feature dataset. 
    It runs through each user and takes the raw data.

    Args:
        folder: Path to folder with .txt files
        window_size: Number of samples per window
        overlap: Fractional overlap (e.g., 0.5 = 50%)

    Returns:
        A DataFrame with features and labels (activity, sensor, etc.)
    """
    folder = Path(folder)
    all_features = []

    for user_folder in folder.iterdir():
        if user_folder.is_dir():
            user_id = user_folder.name
            for path in user_folder.glob("*.txt"):
                try:
                    features = process_accelerometer_file(path, window_size, overlap)
                    features["user"] = user_id  # 💡 add user column here
                    all_features.append(features)
                except Exception as e:
                    print(f"Skipping {path}: {e}")

    return pd.concat(all_features, ignore_index=True)