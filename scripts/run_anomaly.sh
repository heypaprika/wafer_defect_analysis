#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.tasks.detect_anomaly --config configs/anomaly.yaml "$@"
