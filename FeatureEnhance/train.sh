#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/FeatureEnhance/d1_diffaugment.json}
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" train.py --config "$CONFIG"
