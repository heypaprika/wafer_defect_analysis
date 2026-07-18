from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_secom(features_path: str | Path, labels_path: str | Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load UCI SECOM. Features are whitespace-separated with NaNs; labels are {-1, +1}.

    We remap labels to {0, 1} where 1 = fail (the minority, ~7%).
    """
    X = pd.read_csv(features_path, sep=r"\s+", header=None, na_values=["NaN", "nan", ""])
    X.columns = [f"sensor_{i:03d}" for i in range(X.shape[1])]

    lab = pd.read_csv(labels_path, sep=r"\s+", header=None,
                      usecols=[0], names=["label"])
    y = (lab["label"] == 1).astype(int)  # 1 -> fail, -1 -> pass

    assert len(X) == len(y), f"row mismatch: X={len(X)} y={len(y)}"
    return X, y
