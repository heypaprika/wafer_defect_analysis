#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.tasks.analyze_factors --config configs/factor.yaml "$@"
