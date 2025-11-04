from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def run_random_split(X: pd.DataFrame, y: pd.Series, model, seed: int = 42) -> dict[str, Any]:
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)
    return {
        "Split": "random",
        "Accuracy": accuracy_score(yte, yhat),
        "F1 Macro": f1_score(yte, yhat, average="macro"),
        "F1 Weighted": f1_score(yte, yhat, average="weighted"),
    }

def run_groupkfold(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, model, n_splits: int = 5, seed: int = 42
) -> list[dict[str, Any]]:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows: list[dict[str, Any]] = []
    for k, (tr, te) in enumerate(cv.split(X, y, groups)):
        model.fit(X.iloc[tr], y.iloc[tr])
        yhat = model.predict(X.iloc[te])
        rows.append({
            "Split": f"gkfold-{k+1}",
            "Accuracy": accuracy_score(y.iloc[te], yhat),
            "F1 Macro": f1_score(y.iloc[te], yhat, average="macro"),
            "F1 Weighted": f1_score(y.iloc[te], yhat, average="weighted"),
        })
    return rows

def run_louo(
    X: pd.DataFrame, y: pd.Series, users: pd.Series, model,
    save_conf: bool = False, outdir: Path | None = None
) -> list[dict[str, Any]]:
    from sklearn.metrics import confusion_matrix

    rows: list[dict[str, Any]] = []
    class_order = sorted(y.unique().tolist())

    if save_conf:
        assert outdir is not None
        outdir.mkdir(parents=True, exist_ok=True)

    for u in sorted(users.unique()):
        mask = users == u
        model.fit(X[~mask], y[~mask])
        yhat = model.predict(X[mask])

        rows.append({
            "Split": f"LOUO:{u}",
            "Accuracy": accuracy_score(y[mask], yhat),
            "F1 Macro": f1_score(y[mask], yhat, average="macro"),
            "F1 Weighted": f1_score(y[mask], yhat, average="weighted"),
        })

        if save_conf:
            cm = confusion_matrix(y[mask], yhat, labels=class_order)
            cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            np.savetxt(outdir / f"user={u}.counts.csv", cm, fmt="%d", delimiter=",")
            np.savetxt(outdir / f"user={u}.norm.csv", cmn, fmt="%.4f", delimiter=",")
    return rows
