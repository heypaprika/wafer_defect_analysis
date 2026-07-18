"""WM-811K 9-class defect pattern classification.

Pipeline: load -> filter -> resize -> lot-group split -> ResNet + focal loss.
Headline metric is macro-F1 (accuracy is misleading under 85% 'none' prior).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
from src.eval.metrics import dump_json, evaluate_classification
from src.models.cnn import build_cnn
from src.models.losses import FocalLoss, compute_class_weights
from src.utils.config import parse_args_and_load
from src.utils.seed import resolve_device, set_seed
from src.utils.viz import plot_confusion


def _prepare(cfg) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load raw pickle, filter, resize, lot-split. Caches to processed_dir."""
    proc_dir = Path(cfg.data.processed_dir)
    if (proc_dir / "maps.npy").exists():
        d = load_processed(proc_dir)
        return d["maps"], d["labels"], {
            "train": d["idx_train"], "val": d["idx_val"], "test": d["idx_test"],
        }

    print(f"[prepare] loading {cfg.data.raw_path}")
    df = load_wm811k(cfg.data.raw_path)

    classes = list(cfg.labels.classes)
    class_to_idx = {c.lower(): i for i, c in enumerate(classes)}
    df = df[df["failureType"].str.lower().isin(class_to_idx)].reset_index(drop=True)
    df = filter_min_pixels(df, min_pixels=cfg.data.min_pixels)

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
    return maps, labels, {"train": tr, "val": va, "test": te}


def _loader(maps, labels, idx, batch_size, num_workers, shuffle):
    ds = WaferMapDataset(maps[idx], labels[idx])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


def _run_epoch(model, loader, loss_fn, optimizer, device, train: bool):
    model.train(train)
    total, n = 0.0, 0
    preds_all, ys_all = [], []
    for x, y in tqdm(loader, disable=not train, leave=False):
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
        preds_all.append(logits.argmax(1).detach().cpu().numpy())
        ys_all.append(y.detach().cpu().numpy())
    return total / max(n, 1), np.concatenate(ys_all), np.concatenate(preds_all)


def main() -> None:
    cfg = parse_args_and_load()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"[device] {device}")

    maps, labels, splits = _prepare(cfg)
    num_classes = len(cfg.labels.classes)

    train_loader = _loader(maps, labels, splits["train"],
                           cfg.train.batch_size, cfg.train.num_workers, True)
    val_loader = _loader(maps, labels, splits["val"],
                         cfg.train.batch_size, cfg.train.num_workers, False)
    test_loader = _loader(maps, labels, splits["test"],
                          cfg.train.batch_size, cfg.train.num_workers, False)

    model = build_cnn(arch=cfg.model.arch, num_classes=num_classes,
                      in_channels=cfg.model.in_channels,
                      pretrained=cfg.model.pretrained).to(device)

    # Loss: focal with balanced class weights unless overridden
    alpha = None
    if cfg.loss.class_weight == "balanced":
        w = compute_class_weights(labels[splits["train"]], num_classes)
        alpha = torch.from_numpy(w).to(device)
    elif isinstance(cfg.loss.class_weight, list):
        alpha = torch.tensor(cfg.loss.class_weight, dtype=torch.float32, device=device)

    if cfg.loss.type == "focal":
        loss_fn = FocalLoss(alpha=alpha, gamma=cfg.loss.gamma).to(device)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(weight=alpha)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = None
    if cfg.train.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.epochs)

    ckpt_dir = Path(cfg.output.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_f1, patience = -1.0, 0
    for epoch in range(1, cfg.train.epochs + 1):
        tr_loss, _, _ = _run_epoch(model, train_loader, loss_fn, optimizer, device, True)
        va_loss, ytrue, ypred = _run_epoch(model, val_loader, loss_fn, optimizer, device, False)
        from sklearn.metrics import f1_score
        va_f1 = f1_score(ytrue, ypred, average="macro")
        if scheduler is not None:
            scheduler.step()
        print(f"[epoch {epoch:03d}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
              f"val_macroF1={va_f1:.4f}")
        if va_f1 > best_f1:
            best_f1, patience = va_f1, 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"[early-stop] no val improvement for {patience} epochs")
                break

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
    _, ytrue, ypred = _run_epoch(model, test_loader, loss_fn, optimizer, device, False)
    report = evaluate_classification(ytrue, ypred, cfg.labels.classes)
    report["best_val_macro_f1"] = float(best_f1)

    out_dir = Path(cfg.output.report_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(report, out_dir / "metrics.json")
    plot_confusion(np.array(report["confusion"]), cfg.labels.classes,
                   out_dir / "confusion.png")
    pd.DataFrame({"y_true": ytrue, "y_pred": ypred}).to_csv(
        out_dir / "predictions.csv", index=False)

    print(f"[test] macroF1={report['macro_f1']:.4f}  "
          f"balancedAcc={report['balanced_acc']:.4f}  acc={report['accuracy']:.4f}")


if __name__ == "__main__":
    main()
