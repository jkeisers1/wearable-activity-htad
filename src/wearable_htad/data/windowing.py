import pandas as pd


def segment_windows(
    df: pd.DataFrame, window_size: int = 128, overlap: float = 0.5
) -> list[pd.DataFrame]:
    """Segment a time-series DataFrame into overlapping windows.

    Args:
        df: The raw accelerometer DataFrame (must include 'x', 'y', 'z')
        window_size: Number of rows per window
        overlap: Fraction of window to overlap (e.g., 0.5 = 50%)

    Returns:
        A list of DataFrames, each one a window segment.

    """
    stride = int(window_size * (1 - overlap))
    windows = []

    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        window = df.iloc[start:end].copy()
        windows.append(window)

    return windows
