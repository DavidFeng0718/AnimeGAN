# StyleGAN2-ADA Baseline

This directory adds a project-local StyleGAN2-ADA baseline while keeping the
official NVIDIA implementation intact under `upstream/`.

Source: https://github.com/NVlabs/stylegan2-ada-pytorch

## Layout

- `upstream/`: official NVlabs StyleGAN2-ADA PyTorch source copied from GitHub.
- `configs/baseline.json`: baseline experiment config for the local AnimeGAN dataset.
- `configs/animefaces_512.json`: Kaggle animefaces-danbooru 512x512 config.
- `sg2ada/`: project wrapper package for config validation, dataset preparation,
  device handling, training launch, and sample generation.
- `train.py`: backward-compatible JSON-config training entrypoint.
- `generate_samples.py`: backward-compatible batched generation entrypoint.
- `main.py`: modern subcommand CLI (`train`, `generate`, `inspect`).
- `train.sh`: project-style shell entrypoint.
- `runs/`: created at runtime. Official training runs are stored below
  `runs/<experiment_name>/<run-id>-...`.
- `datasets/`: generated StyleGAN2-ADA dataset archives.

`runs/`, generated dataset zip files, and Python caches are ignored by
`StyleGAN2-ADA/.gitignore`.

## Baseline

Default config:

```bash
cd StyleGAN2-ADA
./train.sh
```

Equivalent explicit command:

```bash
cd StyleGAN2-ADA
python train.py --config configs/baseline.json
```

Remote GPU / Colab-style entrypoint:

```bash
cd StyleGAN2-ADA
DATASET_DIR=/path/to/images ./run_colab_baseline_64.sh
```

Dry-run validation:

```bash
cd StyleGAN2-ADA
python train.py --config configs/baseline.json --dry-run
```

Modern CLI equivalent:

```bash
python main.py train --config configs/baseline.json --dry-run
python main.py inspect --config configs/baseline.json
```

Fast startup smoke test:

```bash
cd StyleGAN2-ADA
python train.py --config configs/smoke.json
```

## Kaggle Animefaces 512

The 512 config targets the Kaggle dataset at
`https://www.kaggle.com/datasets/lukexng/animefaces-512x512`. The data itself is
not committed; `datasets/` is ignored because the prepared StyleGAN archive is
large.

Prepare the dataset archive for StyleGAN2-ADA:

```bash
cd StyleGAN2-ADA
scripts/prepare_animefaces_512.sh
```

Colab one-click notebook:

```text
colab/stylegan2_ada_512_one_click.ipynb
```

The notebook keeps the repository, Kaggle download cache, prepared dataset, and
training runs under `/content` while training. When training exits cleanly, it
compresses that `/content` work directory and copies the archive to
`/content/drive/MyDrive/alibaba`.

The script accepts any of these sources:

```bash
SOURCE_DIR=/path/to/extracted/images scripts/prepare_animefaces_512.sh
SOURCE_ZIP=/path/to/animefaces.zip scripts/prepare_animefaces_512.sh
```

If neither source is supplied, it tries the Kaggle CLI first:

```bash
pip install kaggle
export KAGGLE_USERNAME=...
export KAGGLE_KEY=...
scripts/prepare_animefaces_512.sh
```

It also supports `kagglehub` when that package is already installed. The script
creates `datasets/animefaces_512.zip` using the upstream `dataset_tool.py` with a
512x512 center crop and Lanczos resize, then runs a dry-run training validation.

Train after preparation:

```bash
cd StyleGAN2-ADA
python train.py --config configs/animefaces_512.json
```

Or prepare and launch in one command:

```bash
RUN_TRAIN=1 scripts/prepare_animefaces_512.sh
```

The default 512 run uses `cfg=auto`, `aug=ada`, `augpipe=color`, mirroring,
`batch_size=64`, `snap=50`, `allow_tf32=true`, `nhwc=true`, and `kimg=5000`.
`color` avoids the rotation/translation geometry in `bgc`, which can make a
portrait-only face dataset drift into sideways faces. If a smaller Colab GPU
runs out of memory, set `BATCH_SIZE=32`, `16`, or `8` in the notebook or copy
the config and lower `"batch_size"`.

The baseline uses `../dataset`, center-crops images to `64x64`, creates
`datasets/animegan_64.zip`, and trains with:

- `cfg=auto`
- `aug=ada`
- `augpipe=bgc`
- `mirror=true`
- `gpus=1`
- `batch_size=16`
- `kimg=1000`
- `metrics=none`
- `device=auto` (CUDA first, then Apple MPS, then CPU)

For a formal baseline with FID, set `"metrics": "fid50k_full"` in the config.
That will add metric evaluation cost and may download the official feature
detector cache.

Additional wrapper features:

- `"subset": 10000` limits training to a deterministic subset through upstream
  `--subset`.
- `"gamma": 2.5` overrides R1 regularization through upstream `--gamma`.
- `"resume": "latest"` resumes from the newest local `network-snapshot-*.pkl`
  under the experiment run directory.
- `"resume": "latest-if-available"` resumes when a local snapshot exists and
  otherwise starts from scratch.
- `STYLEGAN_RESUME=/path/to/network.pkl` overrides config resume without editing
  JSON.

## Generate Images From a `.pkl`

After training, use the latest `network-snapshot-*.pkl` from the StyleGAN run
directory to export independent PNG samples:

```bash
cd StyleGAN2-ADA
python generate_samples.py \
  --network runs/stylegan2_ada/baseline_64/<run-id>/network-snapshot-001000.pkl \
  --outdir runs/stylegan2_ada/baseline_64/generated_4545 \
  --count 4545 \
  --batch-size 32 \
  --start-seed 0 \
  --trunc 1.0
```

The script writes `generated_000001.png`, `generated_000002.png`, ... plus a
`manifest.json`. Use a fresh output directory unless you intentionally pass
`--overwrite`.

You can also request explicit seeds and a contact sheet:

```bash
python main.py generate --network runs/stylegan2_ada/baseline_64/<run-id>/network-snapshot-001000.pkl \
  --outdir runs/stylegan2_ada/baseline_64/seed_grid \
  --seeds 0-63 \
  --batch-size 32 \
  --grid
```

Latent interpolation between two seeds:

```bash
python generate_samples.py \
  --network runs/stylegan2_ada/baseline_64/<run-id>/network-snapshot-001000.pkl \
  --outdir runs/stylegan2_ada/baseline_64/interp_0_42 \
  --interpolate 0 42 \
  --interpolation-steps 24 \
  --grid
```

For quick visual checks, lower the count:

```bash
python generate_samples.py \
  --network runs/stylegan2_ada/baseline_64/<run-id>/network-snapshot-001000.pkl \
  --outdir runs/stylegan2_ada/baseline_64/preview_64 \
  --count 64
```

## Score With the DCGAN Evaluation Protocol

To compare StyleGAN against the previous DCGAN/FeatureEnhance results, generate
the same number of independent samples and run the existing evaluator:

```bash
cd FeatureEnhance
python evaluate_features.py \
  --config configs/FeatureEnhance/base.json \
  --generated ../StyleGAN2-ADA/runs/stylegan2_ada/baseline_64/generated_4545 \
  --output ../StyleGAN2-ADA/runs/stylegan2_ada/baseline_64/stylegan_eval.json
```

The important metrics are:

- `fid`: lower is better; primary distribution-quality metric.
- `kid_mean`: lower is better; similar to FID and useful for smaller datasets.
- `lpips_diversity_mean`: higher usually means more sample diversity.
- `sampled_near_duplicate_rate` and `exact_duplicate_rate`: lower is better.
- `laplacian_variance_valid_mean`: higher often means sharper images, but can
  also reward noisy artifacts, so treat it as a support metric.

For a fair DCGAN comparison, do not score discriminator-ranked top samples.
Generate independent DCGAN samples with the same config/count:

```bash
cd FeatureEnhance
python GAN_generator.py \
  --config configs/FeatureEnhance/base.json \
  --generator runs/featureEnhance/base_v3/20260619_180759/checkpoints/checkpoint_epoch_300.pt \
  --save-dir runs/featureEnhance/base_v3/20260619_180759/generated_4545 \
  --mode independent \
  --save-num 4545

python evaluate_features.py \
  --config configs/FeatureEnhance/base.json \
  --generated runs/featureEnhance/base_v3/20260619_180759/generated_4545 \
  --output runs/featureEnhance/base_v3/20260619_180759/evaluation/independent_4545.json
```

A StyleGAN result is an improvement if `fid` and `kid_mean` decrease while
duplicate rates stay low and LPIPS diversity does not collapse.

## Runtime Notes

The current target environment is Colab's default Python 3.10+ runtime with
PyTorch 2.7+ and CUDA enabled. Check the active runtime before launching a long
run:

```bash
python -m sg2ada.env --check --require-cuda
```

The official implementation is CUDA-oriented by default. This project copy adds
single-process Apple MPS/CPU support by passing `--device`, disabling CUDA-only
timing/memory paths on non-CUDA devices, forcing fp32 on MPS/CPU, and using the
PyTorch reference implementations for CUDA custom ops.

Device selection is controlled by `"device"` in the JSON config:

- `"auto"`: use CUDA if available, else Apple MPS if available, else CPU.
- `"mps"`: require Apple MPS.
- `"cuda"`: require CUDA.
- `"cpu"`: force CPU.

You can also override without editing JSON:

```bash
STYLEGAN_DEVICE=mps ./train.sh
```

MPS training is much slower than CUDA and only supports `gpus=1`. The wrapper
sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported MPS ops can fall back to CPU.

The copied upstream source includes one local compatibility patch for PyTorch
2.x: `torch_utils/misc.py` calls `Sampler.__init__()` without the removed
`data_source` argument.

If you use a different environment, set:

```bash
PYTHON_BIN=/path/to/python ./train.sh
```
