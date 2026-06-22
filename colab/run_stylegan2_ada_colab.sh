#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZIP_PATH="${1:-"$SCRIPT_DIR/stylegan2_ada_colab_bundle_20260622_151128.zip"}"
WORK_ROOT="${WORK_ROOT:-/content/stylegan2_ada_colab}"
BUNDLE_DIR_NAME="stylegan2_ada_colab_bundle_20260622_151128"
PROJECT_DIR="$WORK_ROOT/$BUNDLE_DIR_NAME/StyleGAN2-ADA"
CONFIG="${CONFIG:-configs/baseline.json}"
AUGPIPE="${AUGPIPE:-color}"

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Package not found: $ZIP_PATH" >&2
  echo "Usage: bash $(basename "$0") /path/to/stylegan2_ada_colab_bundle_20260622_151128.zip" >&2
  exit 1
fi

mkdir -p "$WORK_ROOT"
python - <<PY
import pathlib
import zipfile

zip_path = pathlib.Path("$ZIP_PATH")
work_root = pathlib.Path("$WORK_ROOT")
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(work_root)
print(f"Extracted {zip_path} to {work_root}")
PY

cd "$PROJECT_DIR"

if [[ -f "$PROJECT_DIR/datasets/animegan_64.zip" ]]; then
  export STYLEGAN_DATASET_PATH="$PROJECT_DIR/datasets/animegan_64.zip"
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not available. In Colab, set Runtime > Change runtime type > GPU.")
PY

RUN_CONFIG="$CONFIG"
if [[ -n "${KIMG:-}" || -n "${BATCH:-}" || -n "${SNAP:-}" || -n "${AUGPIPE:-}" ]]; then
  RUN_CONFIG="/content/stylegan2_ada_colab_config.json"
  python - <<PY
import json
from pathlib import Path

src = Path("$CONFIG")
cfg = json.loads(src.read_text())
kimg = "${KIMG:-}"
batch = "${BATCH:-}"
snap = "${SNAP:-}"
augpipe = "${AUGPIPE:-}"
if kimg:
    cfg["kimg"] = int(kimg)
if batch:
    cfg["batch_size"] = int(batch)
if snap:
    cfg["snap"] = int(snap)
if augpipe:
    cfg["augpipe"] = augpipe
Path("$RUN_CONFIG").write_text(json.dumps(cfg, indent=2))
print("Wrote runtime config:", "$RUN_CONFIG")
PY
fi

STYLEGAN_DEVICE=cuda python train.py --config "$RUN_CONFIG"
