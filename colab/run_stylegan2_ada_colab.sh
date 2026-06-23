#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZIP_PATH="${1:-"$SCRIPT_DIR/stylegan2_ada_colab_bundle.zip"}"
WORK_ROOT="${WORK_ROOT:-/content/stylegan2_ada_colab}"
BUNDLE_DIR_NAME="stylegan2_ada_colab_bundle"
PROJECT_DIR="$WORK_ROOT/$BUNDLE_DIR_NAME/StyleGAN2-ADA"
CONFIG="${CONFIG:-configs/cleaned_140k_64.json}"
AUGPIPE="${AUGPIPE:-bgc}"
DATASET_ZIP_NAME="${DATASET_ZIP_NAME:-animegan_cleaned_140k_64.zip}"
DATASET_DEST="$PROJECT_DIR/datasets/$DATASET_ZIP_NAME"

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Package not found: $ZIP_PATH" >&2
  echo "Usage: bash $(basename "$0") /path/to/stylegan2_ada_colab_bundle.zip" >&2
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
mkdir -p "$PROJECT_DIR/datasets"

if [[ ! -f "$DATASET_DEST" ]]; then
  if [[ -n "${DATASET_ZIP_PATH:-}" ]]; then
    echo "Copying dataset zip from DATASET_ZIP_PATH=$DATASET_ZIP_PATH"
    cp "$DATASET_ZIP_PATH" "$DATASET_DEST"
  elif [[ -n "${DATASET_ZIP_URL:-}" ]]; then
    echo "Downloading dataset zip from DATASET_ZIP_URL"
    python - <<PY
from pathlib import Path
from urllib.request import urlretrieve

dest = Path("$DATASET_DEST")
dest.parent.mkdir(parents=True, exist_ok=True)
urlretrieve("$DATASET_ZIP_URL", dest)
print(f"Downloaded dataset to {dest}")
PY
  elif [[ -f "/content/drive/MyDrive/$DATASET_ZIP_NAME" ]]; then
    echo "Copying dataset zip from /content/drive/MyDrive/$DATASET_ZIP_NAME"
    cp "/content/drive/MyDrive/$DATASET_ZIP_NAME" "$DATASET_DEST"
  elif [[ "${MOUNT_DRIVE:-0}" == "1" ]]; then
    python - <<'PY'
from google.colab import drive
drive.mount('/content/drive')
PY
    if [[ -f "/content/drive/MyDrive/$DATASET_ZIP_NAME" ]]; then
      cp "/content/drive/MyDrive/$DATASET_ZIP_NAME" "$DATASET_DEST"
    fi
  fi
fi

if [[ ! -f "$DATASET_DEST" ]]; then
  echo "Dataset zip not found: $DATASET_DEST" >&2
  echo "Provide one of:" >&2
  echo "  DATASET_ZIP_PATH=/content/drive/MyDrive/$DATASET_ZIP_NAME" >&2
  echo "  DATASET_ZIP_URL=https://..." >&2
  echo "  MOUNT_DRIVE=1 with /content/drive/MyDrive/$DATASET_ZIP_NAME" >&2
  exit 1
fi

export STYLEGAN_DATASET_ZIP="$DATASET_DEST"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt --upgrade-strategy only-if-needed

python -m sg2ada.env --check --require-cuda

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
