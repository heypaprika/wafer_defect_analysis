from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def evaluate_classification(y_true, y_pred, labels: Sequence[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": classification_report(
            y_true, y_pred, target_names=list(labels),
            output_dict=True, zero_division=0,
        ),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate_anomaly(scores: np.ndarray, is_anomaly: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(is_anomaly, scores)),
        "n_positive": int(is_anomaly.sum()),
        "n_negative": int((~is_anomaly.astype(bool)).sum()),
        "score_mean_pos": float(scores[is_anomaly.astype(bool)].mean()),
        "score_mean_neg": float(scores[~is_anomaly.astype(bool)].mean()),
    }


def dump_json(report: dict, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(report, f, indent=2)
