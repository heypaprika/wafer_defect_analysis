"""SECOM suspect-factor analysis (KPIV discovery).

GBDT on cleaned sensors, then TreeSHAP for per-feature contribution. Top-K by
mean |SHAP| = suspect factors driving yield loss.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from src.data.load_secom import load_secom
from src.eval.metrics import dump_json
from src.features.selection import preprocess_secom
from src.utils.config import parse_args_and_load
from src.utils.seed import set_seed


def main() -> None:
    cfg = parse_args_and_load()
    set_seed(cfg.seed)

    print(f"[load] {cfg.data.features_path}")
    X_raw, y = load_secom(cfg.data.features_path, cfg.data.labels_path)
    print(f"[load] n={len(X_raw)} sensors={X_raw.shape[1]}  "
          f"positives={int(y.sum())} ({y.mean():.2%})")

    X, meta = preprocess_secom(
        X_raw,
        variance_threshold=cfg.preprocess.drop_variance_threshold,
        imputation=cfg.preprocess.imputation,
        scaler=cfg.preprocess.scaler,
    )
    print(f"[preprocess] kept {X.shape[1]}/{X_raw.shape[1]} sensors after variance filter")

    stratify = y if cfg.split.stratify else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg.split.test_size,
        stratify=stratify, random_state=cfg.seed,
    )

    model = GradientBoostingClassifier(
        n_estimators=cfg.model.n_estimators,
        max_depth=cfg.model.max_depth,
        learning_rate=cfg.model.learning_rate,
        subsample=cfg.model.subsample,
        random_state=cfg.seed,
    )
    model.fit(Xtr, ytr)

    # Basic held-out metrics — useful sanity check, not the deliverable
    from sklearn.metrics import roc_auc_score
    proba = model.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, proba))
    print(f"[gbdt] test AUROC={auc:.4f}")

    # TreeSHAP on the test split — measures contribution to failure
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xte)
    if isinstance(sv, list):  # older shap returned list for binary
        sv = sv[1]

    importance = np.abs(sv).mean(axis=0)
    order = np.argsort(importance)[::-1]
    top_k = int(cfg.explain.top_k)

    suspects = pd.DataFrame({
        "rank": np.arange(1, top_k + 1),
        "sensor": np.asarray(X.columns)[order[:top_k]],
        "mean_abs_shap": importance[order[:top_k]],
    })

    out_dir = Path(cfg.output.report_dir); out_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.output.suspects_csv).parent.mkdir(parents=True, exist_ok=True)
    suspects.to_csv(cfg.output.suspects_csv, index=False)
    dump_json({"test_auroc": auc, "top_k": top_k}, out_dir / "metrics.json")

    if cfg.explain.shap_summary:
        shap.summary_plot(sv, Xte, show=False, max_display=top_k)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary.png", dpi=150)
        plt.close()

    print("\n[top suspect factors]")
    print(suspects.to_string(index=False))


if __name__ == "__main__":
    main()
