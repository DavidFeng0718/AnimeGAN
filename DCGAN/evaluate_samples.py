import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_EPOCHS = [100, 150, 200, 300, 500]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate saved GAN sample grids with lightweight image statistics."
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Run directories, e.g. runs/dataOpt/blur/20260617_113310.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Scan this directory recursively for run folders containing sample/.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional experiment/path filters such as blur noise_filter flip.",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Epoch sample grids to evaluate.",
    )
    parser.add_argument(
        "--output",
        default="sample_quality_metrics.csv",
        help="Combined CSV output path.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Generated image tile size. Defaults to image_size from config.json, or 64.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=2,
        help="Grid padding used by torchvision.utils.make_grid.",
    )
    parser.add_argument(
        "--nrow",
        type=int,
        default=8,
        help="Number of images per grid row used by make_grid.",
    )
    return parser.parse_args()


def load_config(run_dir):
    config_path = run_dir / "config.json"
    if not config_path.exists():
        config_path = run_dir / "config_used.json"
    if not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def discover_runs(args):
    runs = []
    if args.runs:
        runs.extend(Path(path) for path in args.runs)

    if args.root:
        root = Path(args.root)
        runs.extend(path.parent for path in root.rglob("sample") if path.is_dir())

    seen = set()
    unique_runs = []
    for run_dir in runs:
        run_dir = run_dir.resolve()
        if run_dir in seen:
            continue
        seen.add(run_dir)
        if not (run_dir / "sample").is_dir():
            continue
        if args.experiments and not matches_experiment_filter(run_dir, args.experiments):
            continue
        unique_runs.append(run_dir)
    return sorted(unique_runs)


def matches_experiment_filter(run_dir, filters):
    path_text = str(run_dir)
    config = load_config(run_dir)
    candidates = [
        path_text,
        str(config.get("experiment_name", "")),
        str(config.get("augmentation", "")),
    ]
    return any(any(token in candidate for candidate in candidates) for token in filters)


def split_grid(image_path, tile_size, nrow, padding, expected_count):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0

    tiles = []
    if expected_count is None or expected_count <= 0:
        expected_count = nrow * max((arr.shape[0] - padding) // (tile_size + padding), 1)

    for index in range(expected_count):
        row = index // nrow
        col = index % nrow
        x0 = padding + col * (tile_size + padding)
        y0 = padding + row * (tile_size + padding)
        x1 = x0 + tile_size
        y1 = y0 + tile_size
        if y1 > arr.shape[0] or x1 > arr.shape[1]:
            break
        tiles.append(arr[y0:y1, x0:x1, :])

    if not tiles:
        raise ValueError(f"No tiles could be extracted from {image_path}")
    return np.stack(tiles, axis=0)


def rgb_to_gray(images):
    return (
        0.299 * images[..., 0]
        + 0.587 * images[..., 1]
        + 0.114 * images[..., 2]
    )


def laplacian_variance(gray_image):
    center = gray_image[1:-1, 1:-1] * -4.0
    top = gray_image[:-2, 1:-1]
    bottom = gray_image[2:, 1:-1]
    left = gray_image[1:-1, :-2]
    right = gray_image[1:-1, 2:]
    laplacian = center + top + bottom + left + right
    return float(np.var(laplacian))


def pairwise_mean(values):
    if len(values) < 2:
        return 0.0
    return float(np.mean(values))


def image_l2_distances(images):
    flat = images.reshape(images.shape[0], -1)
    distances = []
    for i, j in combinations(range(flat.shape[0]), 2):
        distances.append(float(np.sqrt(np.mean((flat[i] - flat[j]) ** 2))))
    return distances


def color_histograms(images, bins=16):
    histograms = []
    for image in images:
        channel_hists = []
        for channel in range(3):
            hist, _ = np.histogram(image[..., channel], bins=bins, range=(0.0, 1.0))
            hist = hist.astype(np.float32)
            hist /= max(float(hist.sum()), 1.0)
            channel_hists.append(hist)
        histograms.append(np.concatenate(channel_hists))
    return histograms


def histogram_distances(histograms):
    distances = []
    for i, j in combinations(range(len(histograms)), 2):
        distances.append(float(np.abs(histograms[i] - histograms[j]).sum() / 2.0))
    return distances


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std())


def evaluate_grid(image_path, run_dir, config, epoch, args):
    tile_size = args.tile_size or int(config.get("image_size", 64))
    sample_count = int(config.get("sample_count", 64))
    images = split_grid(
        image_path=image_path,
        tile_size=tile_size,
        nrow=args.nrow,
        padding=args.padding,
        expected_count=sample_count,
    )
    gray = rgb_to_gray(images)

    laplacian_values = [laplacian_variance(image) for image in gray]
    gray_contrast_values = gray.reshape(gray.shape[0], -1).std(axis=1)
    brightness_values = gray.reshape(gray.shape[0], -1).mean(axis=1)

    max_rgb = images.max(axis=-1)
    min_rgb = images.min(axis=-1)
    saturation = np.zeros_like(max_rgb)
    np.divide(max_rgb - min_rgb, max_rgb, out=saturation, where=max_rgb > 0)
    saturation_values = saturation.reshape(saturation.shape[0], -1).mean(axis=1)

    color_std_values = images.reshape(images.shape[0], -1, 3).std(axis=1).mean(axis=1)
    l2_distances = image_l2_distances(images)
    hist_distances = histogram_distances(color_histograms(images))
    nearest_l2_values = []
    if len(images) > 1:
        distance_matrix = np.full((len(images), len(images)), np.inf, dtype=np.float32)
        for distance, (i, j) in zip(l2_distances, combinations(range(len(images)), 2)):
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
        nearest_l2_values = distance_matrix.min(axis=1)

    lap_mean, lap_std = summarize(laplacian_values)
    contrast_mean, contrast_std = summarize(gray_contrast_values)
    brightness_mean, brightness_std = summarize(brightness_values)
    saturation_mean, saturation_std = summarize(saturation_values)
    color_std_mean, color_std_std = summarize(color_std_values)
    nearest_l2_mean = float(np.mean(nearest_l2_values)) if len(nearest_l2_values) else 0.0
    low_diversity_pair_ratio = (
        float(np.mean(np.asarray(l2_distances) < 0.05)) if l2_distances else 0.0
    )

    return {
        "run_dir": str(run_dir),
        "experiment_name": config.get("experiment_name", run_dir.parent.name),
        "augmentation": config.get("augmentation", ""),
        "seed": config.get("seed", ""),
        "epoch": epoch,
        "sample_path": str(image_path),
        "image_count": len(images),
        "laplacian_var_mean": lap_mean,
        "laplacian_var_std": lap_std,
        "gray_contrast_mean": contrast_mean,
        "gray_contrast_std": contrast_std,
        "brightness_mean": brightness_mean,
        "brightness_std": brightness_std,
        "saturation_mean": saturation_mean,
        "saturation_std": saturation_std,
        "color_std_mean": color_std_mean,
        "color_std_std": color_std_std,
        "image_l2_diversity": pairwise_mean(l2_distances),
        "color_hist_diversity": pairwise_mean(hist_distances),
        "nearest_l2_mean": nearest_l2_mean,
        "low_diversity_pair_ratio_l2_lt_0_05": low_diversity_pair_ratio,
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError("No sample metrics were generated.")

    fieldnames = list(rows[0].keys())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_run_metrics(run_dir, epochs, tile_size=None, padding=2, nrow=8):
    run_dir = Path(run_dir).resolve()
    config = load_config(run_dir)
    args = argparse.Namespace(tile_size=tile_size, padding=padding, nrow=nrow)
    rows = []
    missing = []
    for epoch in epochs:
        sample_path = run_dir / "sample" / f"fake_images_{epoch}.png"
        if not sample_path.exists():
            missing.append(str(sample_path))
            continue
        rows.append(evaluate_grid(sample_path, run_dir, config, epoch, args))
    return rows, missing


def evaluate_run_samples(
    run_dir,
    epochs=None,
    output_path=None,
    tile_size=None,
    padding=2,
    nrow=8,
):
    run_dir = Path(run_dir).resolve()
    if epochs is None:
        epochs = DEFAULT_EPOCHS
    if output_path is None:
        output_path = run_dir / "sample_quality_metrics.csv"

    rows, missing = collect_run_metrics(
        run_dir=run_dir,
        epochs=epochs,
        tile_size=tile_size,
        padding=padding,
        nrow=nrow,
    )
    write_csv(output_path, rows)
    return rows, missing, Path(output_path)


def main():
    args = parse_args()
    run_dirs = discover_runs(args)
    if not run_dirs:
        raise ValueError("No run directories found. Pass --runs or --root.")

    rows = []
    missing = []
    for run_dir in run_dirs:
        run_rows, run_missing = collect_run_metrics(
            run_dir=run_dir,
            epochs=args.epochs,
            tile_size=args.tile_size,
            padding=args.padding,
            nrow=args.nrow,
        )
        rows.extend(run_rows)
        missing.extend(run_missing)

    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    if missing:
        print(f"Skipped {len(missing)} missing sample grids.")


if __name__ == "__main__":
    main()
