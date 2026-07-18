from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler


def preprocess_secom(
    X: pd.DataFrame,
    variance_threshold: float = 1e-4,
    imputation: str = "median",
    scaler: str = "standard",
) -> Tuple[pd.DataFrame, dict]:
    """Impute → drop near-constant sensors → scale. Returns df + fitted transformers."""
    # Imputation
    if imputation == "median":
        imp = SimpleImputer(strategy="median")
    elif imputation == "mean":
        imp = SimpleImputer(strategy="mean")
    elif imputation == "knn":
        imp = KNNImputer(n_neighbors=5)
    else:
        raise ValueError(f"unknown imputation: {imputation}")
    Xi = imp.fit_transform(X)

    # Variance filter (drop constant / near-constant sensors)
    vt = VarianceThreshold(threshold=variance_threshold)
    Xv = vt.fit_transform(Xi)
    kept = np.asarray(X.columns)[vt.get_support()]

    # Scale
    if scaler == "standard":
        sc = StandardScaler()
    elif scaler == "robust":
        sc = RobustScaler()
    elif scaler == "none":
        sc = None
    else:
        raise ValueError(f"unknown scaler: {scaler}")
    Xs = sc.fit_transform(Xv) if sc is not None else Xv

    out = pd.DataFrame(Xs, columns=kept, index=X.index)
    return out, {"imputer": imp, "vt": vt, "scaler": sc, "kept_columns": list(kept)}
