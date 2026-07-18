from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm


def resize_wafer(wafer_map: np.ndarray, size: int = 64) -> np.ndarray:
    """Nearest-neighbor resize preserves the 3-value semantics (bg/normal/defect)."""
    return cv2.resize(wafer_map.astype("uint8"), (size, size),
                      interpolation=cv2.INTER_NEAREST)


def resize_all(df: pd.DataFrame, size: int = 64) -> np.ndarray:
    out = np.empty((len(df), size, size), dtype=np.uint8)
    for i, wm in enumerate(tqdm(df["waferMap"].values, desc=f"resize->{size}")):
        out[i] = resize_wafer(np.asarray(wm), size=size)
    return out


def filter_min_pixels(df: pd.DataFrame, min_pixels: int = 100) -> pd.DataFrame:
    """Drop wafers whose in-die area is too small — too little signal to learn from."""
    keep = df["waferMap"].apply(lambda wm: int((np.asarray(wm) > 0).sum()) >= min_pixels)
    return df[keep].reset_index(drop=True)


def lot_group_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    group_col: str = "lotName",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split so every lot appears in exactly one of train/val/test.

    Prevents the classic wafer-map leak where sibling wafers from the same lot
    end up on both sides of the split and inflate accuracy.
    """
    if group_col not in df.columns:
        raise KeyError(f"group column '{group_col}' missing from dataframe")

    groups = df[group_col].fillna("__missing__").values

    # 1st split: (train+val) vs test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=groups))

    # 2nd split: within trainval, carve out val
    tv = df.iloc[trainval_idx]
    rel_val = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    tr_rel, va_rel = next(gss2.split(tv, groups=tv[group_col].fillna("__missing__").values))

    train_idx = trainval_idx[tr_rel]
    val_idx = trainval_idx[va_rel]

    # Contract: zero lot overlap between any two splits
    train_lots = set(df.iloc[train_idx][group_col])
    val_lots = set(df.iloc[val_idx][group_col])
    test_lots = set(df.iloc[test_idx][group_col])
    assert not (train_lots & val_lots), "lot leak: train ∩ val"
    assert not (train_lots & test_lots), "lot leak: train ∩ test"
    assert not (val_lots & test_lots), "lot leak: val ∩ test"

    return train_idx, val_idx, test_idx


def save_processed(
    out_dir: str | Path,
    maps: np.ndarray,
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "maps.npy", maps)
    np.save(out / "labels.npy", labels)
    for name, idx in splits.items():
        np.save(out / f"idx_{name}.npy", idx)


def load_processed(out_dir: str | Path):
    out = Path(out_dir)
    return {
        "maps": np.load(out / "maps.npy"),
        "labels": np.load(out / "labels.npy"),
        "idx_train": np.load(out / "idx_train.npy"),
        "idx_val": np.load(out / "idx_val.npy"),
        "idx_test": np.load(out / "idx_test.npy"),
    }
