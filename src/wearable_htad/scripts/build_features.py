# src/wearable_htad/scripts/build_features.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import your actual pipeline function
from wearable_htad.features.pipeline import build_full_feature_dataset


def main():
    """Command-line entry point for feature extraction.

    Example:
        wearable-htad-build --raw-folder data/htad/raw \
                            --output data/processed/features_accel.csv \
                            --window 3.0 --overlap 0.0

    """
    # -----------------------------
    # 1️⃣ Argument parsing
    # -----------------------------
    parser = argparse.ArgumentParser(
        description="Extract accelerometer features from HTAD raw data."
    )

    parser.add_argument(
        "--raw-folder",
        type=Path,
        required=True,
        help="Path to the folder containing user subfolders with raw .txt files.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to the output CSV file to save features."
    )
    parser.add_argument(
        "--window", type=float, default=3.0, help="Window duration in seconds (default: 3.0)."
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Fractional window overlap between 0 and 1 (default: 0.0).",
    )
    parser.add_argument(
        "--sampling-ms",
        type=float,
        default=35.0,
        help="Median sampling interval in milliseconds (default: 35 ms).",
    )

    args = parser.parse_args()

    # -----------------------------
    # 2️⃣ Compute derived parameters
    # -----------------------------
    sampling_rate = 1000 / args.sampling_ms  # Hz
    window_size = int(sampling_rate * args.window)

    # -----------------------------
    # 3️⃣ Run the feature pipeline
    # -----------------------------
    print(f"📥 Loading data from: {args.raw_folder}")
    df = build_full_feature_dataset(args.raw_folder, window_size, args.overlap)

    # Ensure output folder exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Save features
    df.to_csv(args.output, index=False)
    print(f"✅ Saved {len(df)} windows to: {args.output}")
    print(f"🔍 Activities: {df['activity'].nunique()}, Users: {df['user'].nunique()}")


if __name__ == "__main__":
    sys.exit(main())
