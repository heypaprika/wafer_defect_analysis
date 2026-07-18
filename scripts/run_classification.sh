#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.tasks.classify --config configs/classification.yaml "$@"
