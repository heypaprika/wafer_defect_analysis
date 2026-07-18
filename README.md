# Wafer Defect Analysis

End-to-end wafer defect analytics on **WM-811K** (defect-pattern classification, novel-pattern detection) and **SECOM** (yield-loss factor analysis).

## Story

1. **Lot-group split** to prevent inter-lot leakage → honest metrics.
2. **Focal loss + class weighting** to lift minority-class F1 under 85% `None` prior.
3. **Autoencoder on `None` only** → reconstruction error as anomaly score for novel defect patterns (AUROC).
4. **SHAP on SECOM** → rank Key Process Input Variables (KPIVs / "suspect factors").

## Structure

```
configs/          YAML per task — hyperparameters live here, not in code
src/data/         load + preprocess + lot-group split
src/features/     handcrafted wafer features & SECOM feature selection
src/models/       CNN, autoencoder, focal loss
src/tasks/        classify / detect_anomaly / analyze_factors  (CLI entrypoints)
src/eval/         macro-F1, balanced acc, per-class report, confusion
src/utils/        config loader, seed, visualization
notebooks/        EDA & result reports ONLY — no pipeline logic
```

## Data

Place raw files under `data/raw/` (git-ignored):

- `LSWMD.pkl` — WM-811K wafer maps (Kaggle).
- `secom.data` + `secom_labels.data` — UCI SECOM.

## Run

```bash
python -m src.tasks.classify        --config configs/classification.yaml
python -m src.tasks.detect_anomaly  --config configs/anomaly.yaml
python -m src.tasks.analyze_factors --config configs/factor.yaml
```

## Metrics

Accuracy is reported but never the headline. Class imbalance (`None` ~ 85%) makes it meaningless — headline is **macro-F1** and **balanced accuracy**, plus per-class F1 for the eight defect patterns.
