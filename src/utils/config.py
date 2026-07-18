from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_config(path: str | Path) -> SimpleNamespace:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _to_ns(raw)


def parse_args_and_load() -> SimpleNamespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    args = p.parse_args()
    cfg = load_config(args.config)
    cfg._config_path = args.config
    return cfg
