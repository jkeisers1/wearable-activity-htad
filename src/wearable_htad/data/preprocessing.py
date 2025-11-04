from pathlib import Path

import pandas as pd


def load_features_csv(path: str | Path) -> pd.DataFrame:
    """Load the HTAD audio feature CSV with activity labels.

    Args:
        path: Path to the 'features.csv' file

    Returns:
        A pandas DataFrame containing audio features and labels

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    return df


def load_accelerometer_txt(path: str | Path) -> pd.DataFrame:
    """Load the raw accelerometer data from a comma-separated .txt file.

    Also extracts metadata from the filename.

    Args:
        path: Path to the .txt file (e.g., 'user1.txt')

    Returns:
        A pandas DataFrame with sensor readings.

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, header=None)
    df.columns = ["real_timestamp", "x", "y", "z"]

    metadata = parse_filename(path.name)

    for key, value in metadata.items():
        df[key] = value  # add as new columns

    return df


def parse_filename(filename: str) -> dict:
    """Extract metadata from filename with accelerometer data.

    Returns:
        dict with keys: timestamp, sensor, activity.

    """
    name = Path(filename).stem  # remove extension
    parts = name.split("-")
    if len(parts) != 3:
        raise ValueError(f"Unexpected filename format: {filename}")

    return {
        "timestamp": parts[0],
        "sensor": parts[1],
        "activity": parts[2],
    }


def load_all_accelerometer_files(folder: str | Path) -> pd.DataFrame:
    folder = Path(folder)
    files = folder.glob("*.txt")

    all_data = []
    for f in files:
        df = load_accelerometer_txt(f)
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)
