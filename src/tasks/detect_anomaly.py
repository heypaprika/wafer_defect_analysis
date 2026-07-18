"""Novel-pattern detection via one-class autoencoder.

Train on 'none' wafers only. At test time, reconstruction MSE is the anomaly
score; every non-'none' class is treated as the positive (anomaly) class for
AUROC. This mirrors the fab reality that new failure modes appear before
labeled data exists.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.load_wm811k import WaferMapDataset, load_wm811k
from src.data.preprocess import (
    filter_min_pixels,
    load_processed,
    lot_group_split,
    resize_all,
    save_processed,
)
from src.eval.metrics import dump_json, evaluate_anomaly
from src.models.autoencoder import ConvAutoencoder
from src.utils.config import parse_args_and_load
from src.utils.seed import resolve_device, set_seed
from src.utils.viz import plot_score_hist


def _prepare(cfg) -> tuple[np.ndarray, np.ndarray, dict, list[str]]:
    proc_dir = Path(cfg.data.processed_dir)
    if (proc_dir / "maps.npy").exists() and (proc_dir / "classes.npy").exists():
        d = load_processed(proc_dir)
        classes = np.load(proc_dir / "classes.npy", allow_pickle=True).tolist()
        return d["maps"], d["labels"], {
            "train": d["idx_train"], "val": d["idx_val"], "test": d["idx_test"],
        }, classes

    df = load_wm811k(cfg.data.raw_path)
    df = filter_min_pixels(df, min_pixels=cfg.data.min_pixels)

    classes = sorted(df["failureType"].str.lower().unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    labels = df["failureType"].str.lower().map(class_to_idx).values.astype(np.int64)

    tr, va, te = lot_group_split(
        df,
        test_size=cfg.split.test_size,
        val_size=cfg.split.val_size,
        group_col=cfg.split.group_col,
        seed=cfg.seed,
    )
    maps = resize_all(df, size=cfg.data.image_size)
    save_processed(proc_dir, maps, labels, {"train": tr, "val": va, "test": te})
    np.save(proc_dir / "classes.npy", np.array(classes, dtype=object))
    return maps, labels, {"train": tr, "val": va, "test": te}, classes


def _mse_per_sample(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in tqdm(loader, leave=False):
            x = x.to(device)
            recon = model(x)
            err = ((recon - x) ** 2).mean(dim=[1, 2, 3])
            out.append(err.cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    cfg = parse_args_and_load()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    maps, labels, splits, classes = _prepare(cfg)
    train_cls_idx = classes.index(cfg.train_class)

    # Train subset: 'none' only, drawn from the training split
    train_mask = (labels[splits["train"]] == train_cls_idx)
    train_indices = splits["train"][train_mask]
    print(f"[anomaly] train on class='{cfg.train_class}' n={len(train_indices)}")

    train_loader = DataLoader(
        WaferMapDataset(maps[train_indices], labels[train_indices]),
        batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, pin_memory=True,
    )
    val_none_idx = splits["val"][labels[splits["val"]] == train_cls_idx]
    val_loader = DataLoader(
        WaferMapDataset(maps[val_none_idx], labels[val_none_idx]),
        batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        WaferMapDataset(maps[splits["test"]], labels[splits["test"]]),
        batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=True,
    )

    model = ConvAutoencoder(latent_dim=cfg.model.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                                 weight_decay=cfg.train.weight_decay)
    criterion = nn.MSELoss()

    ckpt_dir = Path(cfg.output.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val, patience = float("inf"), 0
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        tr_sum, tr_n = 0.0, 0
        for x, _ in tqdm(train_loader, leave=False):
            x = x.to(device)
            recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tr_sum += float(loss.item()) * x.size(0); tr_n += x.size(0)
        tr_loss = tr_sum / max(tr_n, 1)

        val_err = _mse_per_sample(model, val_loader, device).mean()
        print(f"[epoch {epoch:03d}] train_mse={tr_loss:.5f} val_mse={val_err:.5f}")
        if val_err < best_val:
            best_val, patience = val_err, 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"[early-stop] no val improvement for {patience} epochs")
                break

    # Score the whole test split
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
    scores = _mse_per_sample(model, test_loader, device)
    test_labels = labels[splits["test"]]
    positive_idx = [classes.index(c.lower()) for c in cfg.positive_defect_classes
                    if c.lower() in classes]
    is_anomaly = np.isin(test_labels, positive_idx).astype(int)

    report = evaluate_anomaly(scores, is_anomaly)
    # Per-class mean score to see which defect patterns are easiest to detect
    report["per_class_mean_score"] = {
        classes[c]: float(scores[test_labels == c].mean())
        for c in np.unique(test_labels)
    }

    out_dir = Path(cfg.output.report_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(report, out_dir / "metrics.json")
    plot_score_hist(scores[is_anomaly == 1], scores[is_anomaly == 0],
                    out_dir / "score_hist.png")
    pd.DataFrame({
        "score": scores, "label_idx": test_labels,
        "label": [classes[c] for c in test_labels],
        "is_anomaly": is_anomaly,
    }).to_csv(out_dir / "scores.csv", index=False)

    print(f"[test] AUROC={report['auroc']:.4f}  "
          f"pos={report['n_positive']} neg={report['n_negative']}")


if __name__ == "__main__":
    main()
