from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_wafer(wafer: np.ndarray, title: str = "", ax=None):
    """3-value wafer map: 0 background / 1 normal / 2 defect."""
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(wafer, cmap="viridis", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_confusion(cm: np.ndarray, labels: Sequence[str], out_path: str | Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_score_hist(scores_pos, scores_neg, out_path: str | Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores_neg, bins=60, alpha=0.6, label="normal (none)", density=True)
    ax.hist(scores_pos, bins=60, alpha=0.6, label="defect", density=True)
    ax.set_xlabel("Reconstruction error")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
