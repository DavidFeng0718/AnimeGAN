#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/baseline.json}
if [[ $# -gt 0 ]]; then
  shift
fi
PYTHON_BIN=${PYTHON_BIN:-../FeatureEnhance/.venv/bin/python}

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=${PYTHON_BIN_FALLBACK:-python3}
fi

"$PYTHON_BIN" train.py --config "$CONFIG" "$@"
