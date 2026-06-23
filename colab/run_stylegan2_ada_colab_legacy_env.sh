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
ENV_NAME="${ENV_NAME:-stylegan2ada-py38}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/content/micromamba}"
MAMBA_BIN="$MAMBA_ROOT_PREFIX/bin/micromamba"
PYTORCH_CUDA_WHEEL_INDEX="${PYTORCH_CUDA_WHEEL_INDEX:-https://download.pytorch.org/whl/cu113}"
TORCH_VERSION="${TORCH_VERSION:-1.10.2+cu113}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.11.3+cu113}"

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Package not found: $ZIP_PATH" >&2
  echo "Usage: bash $(basename "$0") /path/to/stylegan2_ada_colab_bundle.zip" >&2
  exit 1
fi

mkdir -p "$WORK_ROOT"
python3 - <<PY
import pathlib
import zipfile

zip_path = pathlib.Path("$ZIP_PATH")
work_root = pathlib.Path("$WORK_ROOT")
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(work_root)
print(f"Extracted {zip_path} to {work_root}")
PY

if [[ ! -x "$MAMBA_BIN" ]]; then
  mkdir -p "$MAMBA_ROOT_PREFIX"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$MAMBA_ROOT_PREFIX" bin/micromamba
fi

if "$MAMBA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if ! "$MAMBA_BIN" run -n "$ENV_NAME" python - <<'PY'
try:
    import torch
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    echo "Existing $ENV_NAME environment has no CUDA PyTorch; recreating it."
    "$MAMBA_BIN" env remove -y -n "$ENV_NAME"
  fi
fi

if ! "$MAMBA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$MAMBA_BIN" create -y -n "$ENV_NAME" \
    -c conda-forge \
    python=3.8 \
    click numpy=1.23 pillow psutil requests scipy tensorboard tqdm ninja
fi

"$MAMBA_BIN" run -n "$ENV_NAME" python -m pip install "setuptools==59.5.0"

"$MAMBA_BIN" run -n "$ENV_NAME" python -m pip install \
  --extra-index-url "$PYTORCH_CUDA_WHEEL_INDEX" \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION"

"$MAMBA_BIN" run -n "$ENV_NAME" python -m pip install pyspng imageio-ffmpeg==0.4.3

cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/datasets"

if [[ ! -f "$DATASET_DEST" ]]; then
  if [[ -n "${DATASET_ZIP_PATH:-}" ]]; then
    echo "Copying dataset zip from DATASET_ZIP_PATH=$DATASET_ZIP_PATH"
    cp "$DATASET_ZIP_PATH" "$DATASET_DEST"
  elif [[ -n "${DATASET_ZIP_URL:-}" ]]; then
    echo "Downloading dataset zip from DATASET_ZIP_URL"
    "$MAMBA_BIN" run -n "$ENV_NAME" python - <<PY
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
    python3 - <<'PY'
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

"$MAMBA_BIN" run -n "$ENV_NAME" python - <<'PY'
import torch

print("python/torch env OK")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not available. In Colab, set Runtime > Change runtime type > GPU.")
PY

RUN_CONFIG="/content/stylegan2_ada_colab_config.json"
"$MAMBA_BIN" run -n "$ENV_NAME" python - <<PY
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

STYLEGAN_DEVICE=cuda "$MAMBA_BIN" run -n "$ENV_NAME" python train.py --config "$RUN_CONFIG"
