#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUNDLE_NAME="${BUNDLE_NAME:-stylegan2_ada_colab_bundle}"
TMP_ROOT="${TMP_ROOT:-/private/tmp/${BUNDLE_NAME}_$$}"
OUT_ZIP="${OUT_ZIP:-$SCRIPT_DIR/stylegan2_ada_colab_bundle.zip}"

mkdir -p "$TMP_ROOT/$BUNDLE_NAME"
mkdir -p "$TMP_ROOT/$BUNDLE_NAME/StyleGAN2-ADA"

rsync -a \
  --exclude ".DS_Store" \
  --exclude ".venv/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude "runs/" \
  --exclude "datasets/*.zip" \
  "$REPO_ROOT/StyleGAN2-ADA/" \
  "$TMP_ROOT/$BUNDLE_NAME/StyleGAN2-ADA/"

mkdir -p "$TMP_ROOT/$BUNDLE_NAME/StyleGAN2-ADA/datasets"

(
  cd "$TMP_ROOT"
  rm -f "$OUT_ZIP"
  zip -qr "$OUT_ZIP" "$BUNDLE_NAME"
)

echo "Wrote $OUT_ZIP"
echo "Bundle root: $BUNDLE_NAME"
