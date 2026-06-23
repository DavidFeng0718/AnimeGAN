#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/animefaces_512.json}"
KAGGLE_DATASET="${KAGGLE_DATASET:-lukexng/animefaces-512x512}"
RAW_DIR="${RAW_DIR:-datasets/animefaces_512_raw}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-datasets/_kaggle_animefaces_512_download}"
DATASET_ZIP="${DATASET_ZIP:-datasets/animefaces_512.zip}"
FORCE="${FORCE:-0}"
RUN_TRAIN="${RUN_TRAIN:-0}"

if [[ "$DATASET_ZIP" = /* ]]; then
  RESOLVED_DATASET_ZIP="$DATASET_ZIP"
else
  RESOLVED_DATASET_ZIP="$PROJECT_DIR/$DATASET_ZIP"
fi

image_probe() {
  find "$1" -type f \( \
    -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o \
    -iname '*.webp' -o -iname '*.bmp' \
  \) -print -quit 2>/dev/null
}

download_with_kaggle_cli() {
  mkdir -p "$DOWNLOAD_DIR"
  kaggle datasets download -d "$KAGGLE_DATASET" -p "$DOWNLOAD_DIR"
  local archive
  archive="$(find "$DOWNLOAD_DIR" -maxdepth 1 -type f -name '*.zip' -print -quit)"
  if [[ -z "$archive" ]]; then
    echo "Kaggle CLI finished, but no zip was found under $DOWNLOAD_DIR" >&2
    exit 1
  fi
  mkdir -p "$DOWNLOAD_DIR/unpacked"
  unzip -q -o "$archive" -d "$DOWNLOAD_DIR/unpacked"
  SOURCE_DIR="$DOWNLOAD_DIR/unpacked"
}

download_with_kagglehub() {
  SOURCE_DIR="$("$PYTHON_BIN" - <<PY
from pathlib import Path
import importlib.util

if importlib.util.find_spec("kagglehub") is None:
    raise SystemExit("kagglehub is not installed")

import kagglehub

path = Path(kagglehub.dataset_download("$KAGGLE_DATASET")).resolve()
print(path)
PY
)"
}

SOURCE_DIR="${SOURCE_DIR:-}"
SOURCE_ZIP="${SOURCE_ZIP:-}"

if [[ -z "$SOURCE_DIR" && -z "$SOURCE_ZIP" ]]; then
  if [[ -d "$RAW_DIR" && -n "$(image_probe "$RAW_DIR")" ]]; then
    SOURCE_DIR="$RAW_DIR"
  elif command -v kaggle >/dev/null 2>&1; then
    download_with_kaggle_cli
  elif "$PYTHON_BIN" -c 'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("kagglehub") else 1)' >/dev/null 2>&1; then
    download_with_kagglehub
  else
    echo "No source images found." >&2
    echo "Provide SOURCE_DIR=/path/to/images, SOURCE_ZIP=/path/to/archive.zip, or install/configure kaggle or kagglehub." >&2
    exit 1
  fi
fi

if [[ -n "$SOURCE_DIR" ]]; then
  SOURCE="$SOURCE_DIR"
elif [[ -n "$SOURCE_ZIP" ]]; then
  SOURCE="$SOURCE_ZIP"
else
  echo "Unable to resolve dataset source." >&2
  exit 1
fi

if [[ ! -e "$SOURCE" ]]; then
  echo "Dataset source does not exist: $SOURCE" >&2
  exit 1
fi

mkdir -p "$(dirname "$RESOLVED_DATASET_ZIP")"
if [[ -f "$RESOLVED_DATASET_ZIP" && "$FORCE" != "1" ]]; then
  echo "Dataset already prepared: $RESOLVED_DATASET_ZIP"
  echo "Set FORCE=1 to rebuild it."
else
  rm -f "$RESOLVED_DATASET_ZIP"
  "$PYTHON_BIN" upstream/dataset_tool.py \
    --source "$SOURCE" \
    --dest "$RESOLVED_DATASET_ZIP" \
    --width 512 \
    --height 512 \
    --transform center-crop \
    --resize-filter lanczos
fi

STYLEGAN_DATASET_PATH="$SOURCE" \
STYLEGAN_DATASET_ZIP="$RESOLVED_DATASET_ZIP" \
  "$PYTHON_BIN" train.py --config "$CONFIG" --dry-run

if [[ "$RUN_TRAIN" == "1" ]]; then
  STYLEGAN_DATASET_PATH="$SOURCE" \
  STYLEGAN_DATASET_ZIP="$RESOLVED_DATASET_ZIP" \
    "$PYTHON_BIN" train.py --config "$CONFIG"
fi
