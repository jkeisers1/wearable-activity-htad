from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def add_row_id(df: pd.DataFrame, user_col: str, label_col: str) -> pd.DataFrame:
    df = df.copy()
    df["row_id"] = (
        df[user_col].astype(str)
        + "_"
        + df[label_col].astype(str)
        + "_"
        + df.groupby([user_col, label_col]).cumcount().astype(str)
    )
    return df

def main() -> int:
    p = argparse.ArgumentParser(description="Fuse audio + accel features into one CSV.")
    p.add_argument("--audio", type=Path, required=True, help="features_audio.csv")
    p.add_argument("--accel", type=Path, required=True, help="features_accel.csv")
    p.add_argument("--out", type=Path, default=Path("data/processed/fusion_features.csv"))
    p.add_argument("--keep-v2", action="store_true", help="Keep v2_ columns (default: drop)")
    args = p.parse_args()

    df_audio = pd.read_csv(args.audio)
    df_accel = pd.read_csv(args.accel)

    # Normalize expected columnss
    if not {"userid", "label"}.issubset(df_audio.columns):
        raise ValueError("Audio CSV must have columns: 'userid', 'label'.")
    if not {"user", "activity"}.issubset(df_accel.columns):
        raise ValueError("Accel CSV must have columns: 'user', 'activity'.")

    df_audio = add_row_id(df_audio, "userid", "label")
    df_accel = add_row_id(df_accel, "user", "activity")

    df = pd.merge(df_audio, df_accel, on="row_id", suffixes=("_audio", "_accel"))

    if not args.keep_v2:
        v2_cols = [c for c in df.columns if c.startswith("v2_")]
        df = df.drop(columns=v2_cols)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✅ Merged: {len(df)} rows, {df.shape[1]} cols → {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
