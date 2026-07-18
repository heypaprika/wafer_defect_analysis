from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss. gamma=0 recovers weighted CE."""

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else torch.tensor([]))
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.alpha if self.alpha.numel() > 0 else None
        ce = F.cross_entropy(logits, target, weight=weight, reduction="none")
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_weights(labels: np.ndarray, num_classes: int,
                          mode: str = "sqrt_inverse") -> np.ndarray:
    """Class weights normalized so they average to 1.

    mode="inverse":       w_c ~ 1 / n_c        (extreme when ratios > 100)
    mode="sqrt_inverse":  w_c ~ 1 / sqrt(n_c)  (moderate; preferred default)
    """
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)  # avoid div/0 for absent classes
    if mode == "inverse":
        raw = 1.0 / counts
    elif mode == "sqrt_inverse":
        raw = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"unknown weight mode: {mode}")
    w = raw * (num_classes / raw.sum())
    return w.astype(np.float32)
