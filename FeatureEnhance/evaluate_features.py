import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from augmentation import get_resize_crop_steps

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EVALUATION_SCHEMA_VERSION = 2


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_project_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def choose_inception_device(device):
    # TorchMetrics accumulates FID statistics in float64, which MPS cannot store.
    return torch.device("cpu") if device.type == "mps" else device


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


class EvaluationImageDataset(Dataset):
    def __init__(self, root, image_size, limit=None):
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {root}")
        self.paths = sorted(
            path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise ValueError(f"No evaluation images found in: {root}")
        self.transform = transforms.Compose(
            get_resize_crop_steps(image_size) + [transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def parse_args():
    parser = argparse.ArgumentParser(description="Compute P0 KID, FID, LPIPS, and diversity metrics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--generated", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def validate_counts(real_dataset, fake_dataset, config):
    if len(real_dataset) != config["evaluation_real_count"]:
        raise ValueError(
            f"Expected {config['evaluation_real_count']} real images, got {len(real_dataset)}."
        )
    if len(fake_dataset) != config["generation_count"]:
        raise ValueError(
            f"Expected {config['generation_count']} generated images, got {len(fake_dataset)}."
        )


def exact_duplicate_rate(paths):
    hashes = []
    for path in paths:
        hashes.append(hashlib.sha256(path.read_bytes()).digest())
    unique = len(set(hashes))
    return (len(hashes) - unique) / len(hashes)


def deterministic_pairs(count, pair_count, seed):
    if count < 2:
        raise ValueError("At least two images are required for diversity evaluation.")
    rng = np.random.default_rng(seed)
    first = rng.integers(0, count, size=pair_count)
    second = rng.integers(0, count - 1, size=pair_count)
    second += second >= first
    return list(zip(first.tolist(), second.tolist()))


def image_laplacian_variance(images, padding=1):
    gray = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=images.dtype,
        device=images.device,
    ).view(1, 1, 3, 3)
    response = functional.conv2d(gray, kernel, padding=padding)
    return response.flatten(1).var(dim=1, unbiased=False)


def image_border_statistics(images, border_width=4, dark_threshold=0.08):
    """Measure dark-border artifacts independently of overall image brightness."""
    gray = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
    height, width = gray.shape[-2:]
    if border_width <= 0 or height <= 2 * border_width or width <= 2 * border_width:
        raise ValueError("border_width must leave a non-empty image interior.")

    border = torch.cat([
        gray[:, :, :border_width, :].flatten(1),
        gray[:, :, -border_width:, :].flatten(1),
        gray[:, :, border_width:-border_width, :border_width].flatten(1),
        gray[:, :, border_width:-border_width, -border_width:].flatten(1),
    ], dim=1)
    interior = gray[:, :, border_width:-border_width, border_width:-border_width].flatten(1)
    border_mean = border.mean(dim=1)
    interior_mean = interior.mean(dim=1)
    return {
        "border_luminance": border_mean,
        "interior_luminance": interior_mean,
        "border_luminance_gap": border_mean - interior_mean,
        "border_dark_pixel_ratio": (border < dark_threshold).float().mean(dim=1),
    }


def compute_basic_metrics(dataset, config):
    loader = DataLoader(dataset, batch_size=config["evaluation_batch_size"], shuffle=False)
    laplacian = []
    laplacian_valid = []
    border_metrics = {
        "border_luminance": [],
        "interior_luminance": [],
        "border_luminance_gap": [],
        "border_dark_pixel_ratio": [],
    }
    for images in loader:
        laplacian.extend(image_laplacian_variance(images).tolist())
        laplacian_valid.extend(image_laplacian_variance(images, padding=0).tolist())
        batch_border_metrics = image_border_statistics(images)
        for name, values in batch_border_metrics.items():
            border_metrics[name].extend(values.tolist())

    pairs = deterministic_pairs(len(dataset), config["lpips_pairs"], config["generation_seed"])
    mse_values = []
    for start in range(0, len(pairs), config["lpips_batch_size"]):
        batch_pairs = pairs[start:start + config["lpips_batch_size"]]
        left = torch.stack([dataset[first] for first, _ in batch_pairs])
        right = torch.stack([dataset[second] for _, second in batch_pairs])
        mse_values.extend(((left - right) ** 2).flatten(1).mean(dim=1).tolist())
    threshold = config["duplicate_mse_threshold"]
    return {
        "laplacian_variance_mean": float(np.mean(laplacian)),
        "laplacian_variance_std": float(np.std(laplacian)),
        "laplacian_variance_valid_mean": float(np.mean(laplacian_valid)),
        "laplacian_variance_valid_std": float(np.std(laplacian_valid)),
        **{
            f"{name}_mean": float(np.mean(values))
            for name, values in border_metrics.items()
        },
        "sampled_pair_mse_mean": float(np.mean(mse_values)),
        "sampled_near_duplicate_rate": float(np.mean(np.asarray(mse_values) <= threshold)),
        "exact_duplicate_rate": exact_duplicate_rate(dataset.paths),
        "sampled_pair_count": len(pairs),
    }


def compute_inception_metrics(real_dataset, fake_dataset, config, device):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except ImportError as exc:
        raise RuntimeError(
            "FID/KID dependencies are unavailable. Install requirements.txt; no substitute metric was written."
        ) from exc

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    kid = KernelInceptionDistance(
        subsets=config["kid_subsets"],
        subset_size=config["kid_subset_size"],
        normalize=True,
    ).to(device)
    total_images = len(real_dataset) + len(fake_dataset)
    processed_images = 0
    next_progress = 5
    started = time.perf_counter()
    print(
        f"Computing FID/KID on {device}: 0/{total_images} images (0%)",
        flush=True,
    )
    for is_real, dataset in ((True, real_dataset), (False, fake_dataset)):
        loader = DataLoader(
            dataset, batch_size=config["evaluation_batch_size"], shuffle=False,
            num_workers=config["num_workers"],
        )
        for images in loader:
            images = images.to(device)
            fid.update(images, real=is_real)
            kid.update(images, real=is_real)
            processed_images += len(images)
            percent = processed_images * 100 // total_images
            if percent >= next_progress or processed_images == total_images:
                elapsed = time.perf_counter() - started
                print(
                    f"Computing FID/KID on {device}: "
                    f"{processed_images}/{total_images} images ({percent}%), "
                    f"{elapsed:.1f}s elapsed",
                    flush=True,
                )
                next_progress = (percent // 5 + 1) * 5
    fid_value = fid.compute().item()
    kid_mean, kid_std = kid.compute()
    return {
        "fid": fid_value,
        "kid_mean": kid_mean.item(),
        "kid_std": kid_std.item(),
    }


def compute_lpips_diversity(dataset, config, device):
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError(
            "LPIPS is unavailable. Install requirements.txt; no substitute metric was written."
        ) from exc
    metric = lpips.LPIPS(net=config["lpips_network"]).to(device).eval()
    pairs = deterministic_pairs(len(dataset), config["lpips_pairs"], config["generation_seed"])
    distances = []
    next_progress = 10
    started = time.perf_counter()
    print(f"Computing LPIPS on {device}: 0/{len(pairs)} pairs (0%)", flush=True)
    with torch.inference_mode():
        for start in range(0, len(pairs), config["lpips_batch_size"]):
            batch_pairs = pairs[start:start + config["lpips_batch_size"]]
            left = torch.stack([dataset[first] for first, _ in batch_pairs]).to(device) * 2 - 1
            right = torch.stack([dataset[second] for _, second in batch_pairs]).to(device) * 2 - 1
            values = metric(left, right, normalize=False).flatten().cpu().tolist()
            distances.extend(values)
            percent = len(distances) * 100 // len(pairs)
            if percent >= next_progress or len(distances) == len(pairs):
                elapsed = time.perf_counter() - started
                print(
                    f"Computing LPIPS on {device}: {len(distances)}/{len(pairs)} "
                    f"pairs ({percent}%), {elapsed:.1f}s elapsed",
                    flush=True,
                )
                next_progress = (percent // 10 + 1) * 10
    return {
        "lpips_diversity_mean": float(np.mean(distances)),
        "lpips_diversity_std": float(np.std(distances)),
        "lpips_pairs": len(distances),
        "lpips_network": config["lpips_network"],
    }


def evaluate(config, generated, output):
    device = choose_device(config["device"])
    inception_device = choose_inception_device(device)
    real_dataset = EvaluationImageDataset(
        resolve_project_path(config["dataset_path"]),
        config["image_size"],
    )
    fake_dataset = EvaluationImageDataset(
        Path(generated).resolve(), config["image_size"]
    )
    validate_counts(real_dataset, fake_dataset, config)
    started = time.perf_counter()
    result = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "preprocessing_version": config.get("preprocessing_version", "legacy_unversioned"),
        "status": "completed",
        "completed_at_utc": utc_now(),
        "real_directory": str(real_dataset.paths[0].parent),
        "generated_directory": str(fake_dataset.paths[0].parent),
        "real_count": len(real_dataset),
        "generated_count": len(fake_dataset),
        "device": str(device),
        "inception_device": str(inception_device),
    }
    result.update(compute_inception_metrics(
        real_dataset, fake_dataset, config, inception_device
    ))
    result.update(compute_lpips_diversity(fake_dataset, config, device))
    result.update(compute_basic_metrics(fake_dataset, config))
    result["seconds"] = time.perf_counter() - started
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"Evaluation produced non-finite metric: {key}={value}")
    write_json(output, result)
    print(f"Evaluation saved: {Path(output).resolve()}")
    return result


def main():
    args = parse_args()
    evaluate(load_config(args.config), args.generated, args.output)


if __name__ == "__main__":
    main()
