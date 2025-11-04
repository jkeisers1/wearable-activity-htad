from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None  # optional dependency

def make_rf() -> Pipeline:
    """RandomForest pipeline with robust defaults."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced_subsample",
        )),
    ])

def make_lgbm(num_classes: int) -> Pipeline:
    """LightGBM pipeline with gentle regularization (requires lightgbm)."""
    if LGBMClassifier is None:
        raise ImportError("lightgbm not installed. Install via `pip install lightgbm`.")
    # No scaler here; LGBM bins features itself. You can add StandardScaler if your data benefits.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LGBMClassifier(
            num_class=num_classes,
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )),
    ])
