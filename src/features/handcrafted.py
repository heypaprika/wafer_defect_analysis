from __future__ import annotations

import numpy as np
from scipy.ndimage import label as cc_label


def defect_density(wafer: np.ndarray) -> float:
    """Fraction of in-die pixels that are defective (label 2)."""
    die = wafer > 0
    if die.sum() == 0:
        return 0.0
    return float((wafer == 2).sum()) / float(die.sum())


def radial_density(wafer: np.ndarray, n_rings: int = 5) -> np.ndarray:
    """Defect density in concentric rings — separates Center / Edge-Ring patterns."""
    h, w = wafer.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = r.max() + 1e-9

    out = np.zeros(n_rings, dtype=np.float32)
    for k in range(n_rings):
        lo, hi = k / n_rings * r_max, (k + 1) / n_rings * r_max
        ring = (r >= lo) & (r < hi) & (wafer > 0)
        n = int(ring.sum())
        out[k] = float(((wafer == 2) & ring).sum()) / n if n else 0.0
    return out


def cluster_stats(wafer: np.ndarray) -> dict[str, float]:
    """Connected-component stats on the defect mask — separates Loc / Scratch / Random."""
    defect_mask = (wafer == 2).astype(np.uint8)
    labels, n = cc_label(defect_mask)
    if n == 0:
        return {"n_clusters": 0, "max_cluster": 0.0, "mean_cluster": 0.0}
    sizes = np.bincount(labels.ravel())[1:]  # drop background
    return {
        "n_clusters": float(n),
        "max_cluster": float(sizes.max()),
        "mean_cluster": float(sizes.mean()),
    }


def extract_features(wafer: np.ndarray, n_rings: int = 5) -> np.ndarray:
    """Concatenate scalar handcrafted features for one wafer map."""
    feats = [defect_density(wafer)]
    feats.extend(radial_density(wafer, n_rings=n_rings).tolist())
    cs = cluster_stats(wafer)
    feats.extend([cs["n_clusters"], cs["max_cluster"], cs["mean_cluster"]])
    return np.asarray(feats, dtype=np.float32)


FEATURE_NAMES = (
    ["defect_density"]
    + [f"ring_{i}" for i in range(5)]
    + ["n_clusters", "max_cluster", "mean_cluster"]
)
