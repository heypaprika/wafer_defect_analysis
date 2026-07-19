# Wafer Defect Analysis

End-to-end wafer defect analytics on **WM-811K** (defect-pattern classification, novel-pattern detection) and **SECOM** (yield-loss factor analysis).

## Results

| Task | Dataset | Headline metric | Files |
|---|---|---|---|
| **Defect classification** (9-class) | WM-811K, 172,950 labeled wafers | **macro-F1 = 0.741**, balanced acc = 0.899 | [metrics.json](reports/classification/metrics.json), [confusion.png](reports/classification/confusion.png) |
| **Novel-pattern detection** (AE) | WM-811K, 147K normal → 25K test | **AUROC = 0.831** | [metrics.json](reports/anomaly/metrics.json), [score_hist.png](reports/anomaly/score_hist.png) |
| **Suspect-factor analysis** (SHAP) | SECOM, 1,567 × 590 sensors | test **AUROC = 0.775**, top-15 KPIVs | [suspect_factors.csv](reports/factor/suspect_factors.csv), [shap_summary.png](reports/factor/shap_summary.png) |

Per-class breakdown highlights that scattered-pattern defects (`Scratch`, `Loc`) remain the hard cases — see [notebook 03](notebooks/03_classification.ipynb).

## Story

1. **Lot-group split** to prevent inter-lot leakage → honest metrics. Same lot never appears in more than one of train/val/test (`assert`-verified). See [notebook 02](notebooks/02_split.ipynb).
2. **Data hygiene**: 638K wafers with empty labels are dropped, not silently mapped to `none`. This alone moved macro-F1 from 0.22 → 0.74.
3. **Sqrt-inverse class weighting** (not raw inverse) + **focal loss (γ=2)** for the 1:1000 minority-vs-majority ratio. Discrete 90° rotation and flip augmentation exploit wafer circular symmetry without pixel interpolation.
4. **Autoencoder trained on truly-labeled `none` only** → reconstruction error is the anomaly score. Detects unseen patterns that a supervised head must misclassify by construction.
5. **SHAP on SECOM** → rank Key Process Input Variables (suspect factors) instead of stopping at accuracy.

## Tutorial notebooks

Sequential walkthrough (open in order):

| # | Notebook | Topic |
|---|---|---|
| 00 | [intro](notebooks/00_intro.ipynb) | Fab / wafer / yield primer; why three tasks |
| 01 | [eda](notebooks/01_eda.ipynb) | Class imbalance, wafer-map samples, SECOM missingness |
| 02 | [split](notebooks/02_split.ipynb) | Why random split leaks; lot-group split demo |
| 03 | [classification](notebooks/03_classification.ipynb) | Accuracy trap, macro-F1, focal loss intuition |
| 04 | [anomaly](notebooks/04_anomaly.ipynb) | Reconstruction-based anomaly detection, AUROC |
| 05 | [factor](notebooks/05_factor.ipynb) | SHAP for KPIV ranking |

Each notebook: **concept → code → interpretation**, with `src/` imports (not reimplementation).

## Structure

```
configs/          YAML per task — hyperparameters live here, not in code
src/data/         load + preprocess + lot-group split
src/features/     handcrafted wafer features & SECOM feature selection
src/models/       CNN, autoencoder, focal loss
src/tasks/        classify / detect_anomaly / analyze_factors  (CLI entrypoints)
src/eval/         macro-F1, balanced acc, per-class report, confusion
src/utils/        config loader, seed, visualization
notebooks/        six-chapter tutorial (00_intro → 05_factor)
reports/          per-task metrics.json, csv, and diagnostic plots
```

## Data

Place raw files under `data/raw/` (git-ignored):

- `LSWMD.pkl` — WM-811K wafer maps ([Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)).
- `secom.data` + `secom_labels.data` — [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom).

The WM-811K pickle needs three compat shims to load on modern Python — all handled in [src/data/load_wm811k.py](src/data/load_wm811k.py).

## Run

```bash
python -m src.tasks.classify        --config configs/classification.yaml
python -m src.tasks.detect_anomaly  --config configs/anomaly.yaml
python -m src.tasks.analyze_factors --config configs/factor.yaml
```

Or via wrappers: `bash scripts/run_classification.sh`, `run_anomaly.sh`, `run_factor.sh`.

First run of each task caches resized wafer maps under `data/processed/` (git-ignored). Re-runs load from cache.

## Metrics

Accuracy is reported but never the headline. Under `none ~ 85%` prior, a trivial "always none" classifier scores 0.85 accuracy while catching zero defects. Headline metrics:

- **macro-F1** — treats every class equally; the honest measure under imbalance
- **balanced accuracy** — average per-class recall
- **per-class F1** — reveals which defect types are actually being caught
- **AUROC** (anomaly) — threshold-free separation between normal and defect score distributions

## Environment

```bash
pip install -r requirements.txt
```

Tested on Python 3.11 + PyTorch 2.2, CUDA 12.1 (RunPod RTX 3090). CPU-only will work but classification/anomaly training are impractically slow (~hours/epoch); use a GPU pod.
