
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wearable_htad.features.selection import select_features
from wearable_htad.modeling.cv import run_groupkfold, run_louo, run_random_split
from wearable_htad.modeling.models import make_lgbm, make_rf

LABEL, GROUP = "activity", "user"

def parse_args():
    p = argparse.ArgumentParser(description="Train models with different CV and modalities.")
    p.add_argument("--data", type=Path, required=True, help="Path to features CSV (fusion or single-modality).")
    p.add_argument("--model", choices=["rf", "lgbm", "all"], default="rf")
    p.add_argument("--modality", choices=["accel", "audio", "fusion", "all"], default="fusion")
    p.add_argument("--cv", choices=["random", "groupkfold", "louo"], default="louo")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=Path, default=Path("outputs/metrics"))
    p.add_argument("--save-confusion", action="store_true")
    return p.parse_args()

def get_models(which: str, n_classes: int):
    out = []
    if which in ("rf", "all"):
        out.append(("rf", make_rf()))
    if which in ("lgbm", "all"):
        out.append(("lgbm", make_lgbm(n_classes)))
    return out

def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.data)
    y = df[LABEL].astype(str)
    users = df[GROUP].astype(str)

    modalities = [args.modality] if args.modality != "all" else ["accel", "audio", "fusion"]
    all_rows = []

    for mod in modalities:
        X = select_features(df, modality=mod)
        n_classes = y.nunique()
        for name, model in get_models(args.model, n_classes):
            if args.cv == "random":
                rows = [run_random_split(X, y, model, seed=args.seed)]
            elif args.cv == "groupkfold":
                rows = run_groupkfold(X, y, users, model, n_splits=args.n_splits, seed=args.seed)
            else:  # LOUO
                conf_dir = Path("outputs/confusion") / f"{name}_{mod}" if args.save_confusion else None
                rows = run_louo(X, y, users, model, save_conf=args.save_confusion, outdir=conf_dir)
            for r in rows:
                r.update({"Model": name, "Modality": mod, "CV": args.cv})
            all_rows.extend(rows)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_csv = args.outdir / f"metrics_{args.model}_{args.modality}_{args.cv}.csv"
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"✅ Saved metrics to {out_csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
