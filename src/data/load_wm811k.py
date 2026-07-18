from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_wm811k(pkl_path: str | Path) -> pd.DataFrame:
    """Load WM-811K pickle and normalize label / column names.

    Returns a DataFrame with at least: waferMap, failureType, lotName.
    failureType is coerced to a python string; 'none' is normalized lowercase.
    """
    df = pd.read_pickle(pkl_path)

    # WM-811K columns sometimes come nested as 1-element arrays
    def _flatten(cell):
        if isinstance(cell, (list, np.ndarray)) and len(cell) == 1:
            return cell[0]
        return cell

    for col in ("failureType", "trainTestLabel", "lotName"):
        if col in df.columns:
            df[col] = df[col].apply(_flatten)

    # Keep only rows with a known failureType label
    df = df[df["failureType"].apply(lambda x: isinstance(x, str) and len(x) > 0)]
    df["failureType"] = df["failureType"].str.strip()
    df.loc[df["failureType"].str.lower() == "none", "failureType"] = "none"

    return df.reset_index(drop=True)


class WaferMapDataset(Dataset):
    """Torch Dataset over pre-resized wafer maps (uint8 in {0,1,2})."""

    def __init__(
        self,
        maps: np.ndarray,        # (N, H, W)
        labels: Sequence[int],
        transform=None,
    ) -> None:
        assert maps.ndim == 3, f"expected (N,H,W), got {maps.shape}"
        self.maps = maps
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        x = torch.from_numpy(self.maps[i]).float().unsqueeze(0) / 2.0  # scale to [0,1]
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.labels[i])
