#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$REPO_ROOT/StyleGAN2-ADA"
CONFIG="${CONFIG:-configs/cleaned_140k_64.json}"
DATASET_ZIP_NAME="${DATASET_ZIP_NAME:-animegan_cleaned_140k_64.zip}"
DRIVE_DATASET_PATH="${DRIVE_DATASET_PATH:-/content/drive/MyDrive/$DATASET_ZIP_NAME}"
CONTENT_DATASET_PATH="${CONTENT_DATASET_PATH:-/content/$DATASET_ZIP_NAME}"
AUGPIPE="${AUGPIPE:-bgc}"

python - <<'PY'
from google.colab import drive
drive.mount('/content/drive')
PY

if [[ ! -f "$DRIVE_DATASET_PATH" ]]; then
  echo "Dataset zip not found in Google Drive root: $DRIVE_DATASET_PATH" >&2
  echo "Set DATASET_ZIP_NAME=your_file.zip or DRIVE_DATASET_PATH=/content/drive/MyDrive/your_file.zip" >&2
  exit 1
fi

echo "Copying dataset to $CONTENT_DATASET_PATH"
cp "$DRIVE_DATASET_PATH" "$CONTENT_DATASET_PATH"

cd "$PROJECT_DIR"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --upgrade-strategy only-if-needed
python -m sg2ada.env --check --require-cuda

RUN_CONFIG="/content/stylegan2_ada_colab_config.json"
python - <<PY
import json
from pathlib import Path

src = Path("$CONFIG")
cfg = json.loads(src.read_text())
cfg["dataset_zip"] = "$CONTENT_DATASET_PATH"
cfg["prepare_dataset"] = False
cfg["force_prepare_dataset"] = False
cfg["device"] = "cuda"
kimg = "${KIMG:-}"
batch = "${BATCH:-}"
snap = "${SNAP:-}"
augpipe = "${AUGPIPE:-}"
resume = "${RESUME:-}"
if kimg:
    cfg["kimg"] = int(kimg)
if batch:
    cfg["batch_size"] = int(batch)
if snap:
    cfg["snap"] = int(snap)
if augpipe:
    cfg["augpipe"] = augpipe
if resume:
    cfg["resume"] = resume
Path("$RUN_CONFIG").write_text(json.dumps(cfg, indent=2))
print("Wrote runtime config:", "$RUN_CONFIG")
PY

STYLEGAN_DEVICE=cuda STYLEGAN_DATASET_ZIP="$CONTENT_DATASET_PATH" python train.py --config "$RUN_CONFIG"
