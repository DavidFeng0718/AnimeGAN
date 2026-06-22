#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -m pip install -r requirements.txt

if [ -n "${DATASET_DIR:-}" ]; then
  mkdir -p ../DCGAN
  rm -f ../DCGAN/dataset_cleaned
  ln -s "$DATASET_DIR" ../DCGAN/dataset_cleaned
elif [ -d ./dataset_cleaned ] && [ ! -e ../DCGAN/dataset_cleaned ]; then
  mkdir -p ../DCGAN
  ln -s ../FeatureEnhance/dataset_cleaned ../DCGAN/dataset_cleaned
fi

python train.py --config configs/FeatureEnhance/d1_noise_label_autocontrast_64.json
