#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -m pip install -r requirements.txt

if [ -n "${DATASET_DIR:-}" ]; then
  export STYLEGAN_DATASET_PATH="$DATASET_DIR"
fi

python train.py --config configs/baseline.json
