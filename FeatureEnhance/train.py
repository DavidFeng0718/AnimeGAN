import argparse
import csv
import json
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from augmentation import get_transform
from create_dataset import My_dataset, save_img
from diff_augment import diff_augment, parse_diff_augment_policy
from evaluate_samples import DEFAULT_EPOCHS, evaluate_run_samples
from model64 import Generator as Generator64, Discriminator as Discriminator64
from model64_se import Generator as Generator64SE
from model64_sn import Discriminator as Discriminator64SN
from model64_sobel import Generator as Generator64Sobel
from model64_dwt import Generator as Generator64DWT
from model512 import Generator as Generator512, Discriminator as Discriminator512


REQUIRED_CONFIG_KEYS = [
    "experiment_name",
    "image_size",
    "batch_size",
    "epochs",
    "noise_dim",
    "g_lr",
    "d_lr",
    "beta1",
    "beta2",
    "dataset_path",
    "num_workers",
    "augmentation",
]

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = "configs/FeatureEnhance/d1_diffaugment.json"
IMAGE_SIZE_MODEL_REGISTRY = {
    64: (Generator64, Discriminator64, "model64.py"),
    512: (Generator512, Discriminator512, "model512.py"),
}
MODEL_VARIANT_REGISTRY = {
    "base": (Generator64, Discriminator64, "model64.py", 64),
    "a1_se": (Generator64SE, Discriminator64, "model64_se.py", 64),
    "s1_spectralnorm": (Generator64, Discriminator64SN, "model64_sn.py", 64),
    "e1_sobel": (Generator64Sobel, Discriminator64, "model64_sobel.py", 64),
    "f1_haar_dwt": (Generator64DWT, Discriminator64, "model64_dwt.py", 64),
}
CURRENT_PREPROCESSING_VERSION = "reflect_rotation_p60_v3"
FEATURE_BASE_REQUIRED_KEYS = {
    "schema_version", "preprocessing_version", "experiment_name", "model_variant", "dataset_name",
    "dataset_path", "image_size", "image_channels", "noise_dim", "batch_size",
    "epochs", "seed", "num_workers", "device", "g_lr", "d_lr", "beta1",
    "beta2", "real_label", "fake_label", "g_steps", "d_train_every",
    "instance_noise_std", "instance_noise_decay_epochs", "augmentation",
    "rotation_degrees", "rotation_probability", "color_jitter_brightness", "color_jitter_contrast",
    "color_jitter_saturation", "color_jitter_hue",
    "horizontal_flip_probability", "checkpoint_interval", "sample_interval", "fixed_noise_seed",
    "diff_augment_policy", "diff_augment_translation_ratio", "diff_augment_cutout_ratio",
    "sample_count", "evaluation_epochs", "generation_count",
    "generation_batch_size", "generation_seed", "evaluation_real_count",
    "evaluation_batch_size", "kid_subsets", "kid_subset_size", "lpips_pairs",
    "lpips_batch_size", "lpips_network", "duplicate_mse_threshold", "run_root",
}
RESUME_COMPATIBILITY_KEYS = {
    "model_variant", "dataset_path", "image_size", "image_channels", "noise_dim",
    "batch_size", "g_lr", "d_lr", "beta1", "beta2", "real_label", "fake_label",
    "g_steps", "d_train_every", "instance_noise_std", "instance_noise_decay_epochs",
    "augmentation", "rotation_degrees", "rotation_probability", "color_jitter_brightness",
    "color_jitter_contrast", "color_jitter_saturation", "color_jitter_hue",
    "horizontal_flip_probability", "preprocessing_version",
    "diff_augment_policy", "diff_augment_translation_ratio", "diff_augment_cutout_ratio",
}


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train DCGAN with a JSON experiment config.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Experiment JSON config (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument("--resume", default=None, help="Optional full checkpoint to restore.")
    return parser.parse_args(args)


def load_config(config_path):
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as fp:
        config = json.load(fp)

    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing_keys:
        raise ValueError(f"Missing config keys: {', '.join(missing_keys)}")

    validate_config(config)
    return config


def validate_config(config):
    image_size = config["image_size"]
    if image_size not in IMAGE_SIZE_MODEL_REGISTRY:
        supported_sizes = ", ".join(str(size) for size in sorted(IMAGE_SIZE_MODEL_REGISTRY))
        raise ValueError(f"Unsupported image_size {image_size}. Supported sizes: {supported_sizes}")

    model_variant = config.get("model_variant")
    if model_variant is not None:
        if model_variant not in MODEL_VARIANT_REGISTRY:
            supported = ", ".join(sorted(MODEL_VARIANT_REGISTRY))
            raise ValueError(f"Unsupported model_variant {model_variant}. Supported variants: {supported}")
        expected_size = MODEL_VARIANT_REGISTRY[model_variant][3]
        if image_size != expected_size:
            raise ValueError(f"model_variant {model_variant} requires image_size={expected_size}.")
    if model_variant in {
        "base", "a1_se", "s1_spectralnorm", "e1_sobel", "f1_haar_dwt"
    }:
        missing = sorted(FEATURE_BASE_REQUIRED_KEYS - config.keys())
        if missing:
            raise ValueError(f"Feature experiment must explicitly define: {', '.join(missing)}")
        if config["generation_count"] < 4545 or config["evaluation_real_count"] < 4545:
            raise ValueError("Feature Base requires at least 4545 generated and real samples.")
        if config["schema_version"] != 3:
            raise ValueError("Feature experiments must use schema_version=3 for probabilistic rotation.")
        if config["preprocessing_version"] != CURRENT_PREPROCESSING_VERSION:
            raise ValueError(
                f"Unsupported preprocessing_version {config['preprocessing_version']!r}; "
                f"expected {CURRENT_PREPROCESSING_VERSION!r}."
            )
        if config["image_channels"] != 3:
            raise ValueError("Current generators and discriminators require image_channels=3.")
        if not isinstance(config["rotation_probability"], (int, float)):
            raise ValueError("Config 'rotation_probability' must be numeric.")
        if not 0.0 <= config["rotation_probability"] <= 1.0:
            raise ValueError("Config 'rotation_probability' must be in [0, 1].")
        parse_diff_augment_policy(config["diff_augment_policy"])
        translation_ratio = config["diff_augment_translation_ratio"]
        cutout_ratio = config["diff_augment_cutout_ratio"]
        if not isinstance(translation_ratio, (int, float)) or not 0.0 <= translation_ratio <= 0.5:
            raise ValueError("Config 'diff_augment_translation_ratio' must be in [0, 0.5].")
        if not isinstance(cutout_ratio, (int, float)) or not 0.0 <= cutout_ratio <= 1.0:
            raise ValueError("Config 'diff_augment_cutout_ratio' must be in [0, 1].")

    int_keys = ["image_size", "batch_size", "epochs", "noise_dim"]
    for key in int_keys:
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"Config '{key}' must be a positive integer.")

    if not isinstance(config["num_workers"], int) or config["num_workers"] < 0:
        raise ValueError("Config 'num_workers' must be a non-negative integer.")

    for key in ["g_lr", "d_lr"]:
        if not isinstance(config[key], (int, float)) or config[key] <= 0:
            raise ValueError(f"Config '{key}' must be a positive number.")

    for key in ["beta1", "beta2"]:
        if not isinstance(config[key], (int, float)):
            raise ValueError(f"Config '{key}' must be numeric.")

    if not 0 <= config["beta1"] < 1 or not 0 <= config["beta2"] < 1:
        raise ValueError("Adam beta1 and beta2 must be in [0, 1).")

    if "seed" in config and (not isinstance(config["seed"], int) or config["seed"] < 0):
        raise ValueError("Config 'seed' must be a non-negative integer.")

    for key in [
        "g_steps",
        "d_train_every",
        "instance_noise_decay_epochs",
        "checkpoint_interval",
        "sample_interval",
        "sample_count",
        "fixed_noise_seed",
        "generation_count",
        "generation_batch_size",
        "evaluation_real_count",
        "evaluation_batch_size",
        "kid_subsets",
        "kid_subset_size",
        "lpips_pairs",
        "lpips_batch_size",
    ]:
        if key in config and (not isinstance(config[key], int) or config[key] <= 0):
            raise ValueError(f"Config '{key}' must be a positive integer.")

    for key in ["real_label", "fake_label", "instance_noise_std"]:
        if key in config and not isinstance(config[key], (int, float)):
            raise ValueError(f"Config '{key}' must be numeric.")

    real_label = config.get("real_label", 0.9)
    fake_label = config.get("fake_label", 0.1)
    if not 0 <= fake_label < real_label <= 1:
        raise ValueError("Config labels must satisfy 0 <= fake_label < real_label <= 1.")

    if config.get("instance_noise_std", 0.0) < 0:
        raise ValueError("Config 'instance_noise_std' must be non-negative.")

    if "evaluation_epochs" in config:
        evaluation_epochs = config["evaluation_epochs"]
        if (
            not isinstance(evaluation_epochs, list)
            or not evaluation_epochs
            or any(not isinstance(epoch, int) or epoch <= 0 for epoch in evaluation_epochs)
        ):
            raise ValueError("Config 'evaluation_epochs' must be a non-empty list of positive integers.")

    if "kid_subset_size" in config:
        available_count = min(config["generation_count"], config["evaluation_real_count"])
        if config["kid_subset_size"] > available_count:
            raise ValueError("Config 'kid_subset_size' cannot exceed the available evaluation images.")


def get_model_classes(config):
    model_variant = config.get("model_variant")
    if model_variant is not None:
        Generator, Discriminator, model_file, _ = MODEL_VARIANT_REGISTRY[model_variant]
        return Generator, Discriminator, model_file
    return IMAGE_SIZE_MODEL_REGISTRY[config["image_size"]]


def resolve_project_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()


def create_run_dir(experiment_name, run_root="runs"):
    run_root = resolve_project_path(run_root)
    experiment_dir = run_root / experiment_name
    while True:
        run_dir = experiment_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            time.sleep(1)

    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "sample").mkdir(parents=True, exist_ok=True)
    (run_dir / "source_code").mkdir(parents=True, exist_ok=True)
    return run_dir


def log_message(message, log_fp):
    print(message)
    log_fp.write(message + "\n")
    log_fp.flush()


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=4)


def save_config(run_dir, config, dataset_size, dataset_path, config_path, model_file):
    config_to_save = dict(config)
    config_to_save.update({
        "optimizer": "Adam",
        "loss": "BCEWithLogitsLoss",
        "dataset_size": dataset_size,
        "resolved_dataset_path": str(dataset_path),
        "config_file": str(config_path),
        "model_file": model_file,
    })
    save_json(run_dir / "config.json", config_to_save)


def save_config_used(run_dir, config_path):
    shutil.copy2(config_path, run_dir / "config_used.json")


def save_environment(run_dir, device):
    gpu_name = "N/A"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        gpu_name = "Apple MPS"

    environment = [
        f"Python {platform.python_version()}",
        f"Torch {torch.__version__}",
        f"CUDA {torch.version.cuda if torch.version.cuda else 'N/A'}",
        "",
        f"MPS availability: {torch.backends.mps.is_available()}",
        f"Device: {device}",
        f"GPU: {gpu_name}",
        "",
        f"OS: {platform.platform()}",
    ]
    with open(run_dir / "environment.txt", "w", encoding="utf-8") as fp:
        fp.write("\n".join(environment) + "\n")
    try:
        packages = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        packages = f"Unavailable: {exc}\n"
    with open(run_dir / "requirements_frozen.txt", "w", encoding="utf-8") as fp:
        fp.write(packages)


def save_source_code(run_dir, model_file):
    source_dir = run_dir / "source_code"
    source_files = [
        "train.py", "augmentation.py", "main.py", "create_dataset.py",
        "evaluate_samples.py", "evaluate_features.py", "GAN_generator.py",
        "feature_modules.py", "diff_augment.py", model_file,
    ]

    for source_file in source_files:
        source_path = PROJECT_DIR / source_file
        if source_path.exists():
            shutil.copy2(source_path, source_dir / source_path.name)


def save_git_commit(run_dir):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_DIR,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        commit = "Unavailable"

    with open(run_dir / "git_commit.txt", "w", encoding="utf-8") as fp:
        fp.write(commit + "\n")
    try:
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=PROJECT_DIR,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception as exc:
        status = f"Unavailable: {exc}\n"
    with open(run_dir / "git_status.txt", "w", encoding="utf-8") as fp:
        fp.write(status)


def save_model_architecture(run_dir, generator, discriminator):
    with open(run_dir / "model_architecture.txt", "w", encoding="utf-8") as fp:
        fp.write("Generator\n")
        fp.write(str(generator))
        fp.write("\n\nDiscriminator\n")
        fp.write(str(discriminator))
        fp.write("\n")


def save_parameter_counts(run_dir, generator, discriminator):
    counts = {
        "generator": sum(parameter.numel() for parameter in generator.parameters()),
        "discriminator": sum(parameter.numel() for parameter in discriminator.parameters()),
    }
    counts["total"] = counts["generator"] + counts["discriminator"]
    save_json(run_dir / "parameter_counts.json", counts)
    return counts


def draw_line_chart(path, x_values, series, y_label):
    width, height = 1000, 600
    margin_left, margin_right, margin_top, margin_bottom = 90, 40, 40, 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    all_y_values = [value for values in series.values() for value in values]
    if not x_values or not all_y_values:
        return

    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(all_y_values), max(all_y_values)
    if min_x == max_x:
        min_x -= 1
        max_x += 1
    if min_y == max_y:
        min_y -= 1
        max_y += 1

    def scale_x(value):
        return int(round(margin_left + (value - min_x) / (max_x - min_x) * plot_width))

    def scale_y(value):
        return int(round(margin_top + (max_y - value) / (max_y - min_y) * plot_height))

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.line((margin_left, margin_top, margin_left, height - margin_bottom), fill="black", width=2)
    draw.line((margin_left, height - margin_bottom, width - margin_right, height - margin_bottom), fill="black", width=2)

    for tick in range(6):
        y = margin_top + tick * plot_height / 5
        value = max_y - tick * (max_y - min_y) / 5
        draw.line((margin_left - 5, y, width - margin_right, y), fill="#e6e6e6")
        draw.text((10, y - 7), f"{value:.4f}", fill="black")

    for tick in range(6):
        x = margin_left + tick * plot_width / 5
        value = min_x + tick * (max_x - min_x) / 5
        draw.line((x, height - margin_bottom, x, height - margin_bottom + 5), fill="black")
        draw.text((x - 12, height - margin_bottom + 12), f"{value:.0f}", fill="black")

    draw.text((width // 2 - 20, height - 35), "Epoch", fill="black")
    draw.text((10, 15), y_label, fill="black")

    for series_index, (label, values) in enumerate(series.items()):
        color = colors[series_index % len(colors)]
        points = [(scale_x(x), scale_y(y)) for x, y in zip(x_values, values)]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        else:
            draw.line(points, fill=color, width=3)
        legend_y = margin_top + series_index * 24
        legend_x = width - margin_right - 220
        draw.line((legend_x, legend_y + 8, legend_x + 30, legend_y + 8), fill=color, width=3)
        draw.text((legend_x + 38, legend_y), label, fill="black")

    image.save(path)


def plot_training_curves(run_dir, epoch_metrics):
    if not epoch_metrics:
        return

    epochs = [item["epoch"] for item in epoch_metrics]
    draw_line_chart(
        run_dir / "plots" / "loss_curve.png",
        epochs,
        {
            "Generator Loss": [item["avg_g_loss"] for item in epoch_metrics],
            "Discriminator Loss": [item["avg_d_loss"] for item in epoch_metrics],
        },
        "Loss",
    )
    draw_line_chart(
        run_dir / "plots" / "discriminator_curve.png",
        epochs,
        {
            "D_real": [item["avg_D_real"] for item in epoch_metrics],
            "D_fake": [item["avg_D_fake"] for item in epoch_metrics],
        },
        "Discriminator Output",
    )
    draw_line_chart(
        run_dir / "plots" / "gap_curve.png",
        epochs,
        {"D_real - D_fake": [item["avg_D_gap"] for item in epoch_metrics]},
        "D_gap",
    )


def get_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_peak_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    return None


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_fixed_noise(sample_count, noise_dim, seed, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(sample_count, noise_dim, 1, 1, generator=generator).to(device)


def set_requires_grad(module, requires_grad):
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def capture_rng_state(dataloader_generator=None):
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if dataloader_generator is not None:
        state["dataloader"] = dataloader_generator.get_state()
    return state


def restore_rng_state(state, dataloader_generator=None):
    if not state:
        return False
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    if dataloader_generator is not None and "dataloader" in state:
        dataloader_generator.set_state(state["dataloader"].cpu())
    return True


def validate_resume_config(checkpoint_config, current_config):
    if not checkpoint_config or current_config is None:
        return
    mismatches = [
        key for key in sorted(RESUME_COMPATIBILITY_KEYS)
        if checkpoint_config.get(key) != current_config.get(key)
    ]
    if mismatches:
        raise ValueError(
            "Resume checkpoint is incompatible with the current config: "
            + ", ".join(mismatches)
        )


def get_instance_noise_std(config, epoch):
    std = config.get("instance_noise_std", 0.0)
    if std <= 0:
        return 0.0

    decay_epochs = config.get("instance_noise_decay_epochs", config["epochs"])
    progress = min(epoch / decay_epochs, 1.0)
    return std * (1.0 - progress)


def add_instance_noise(images, std):
    if std <= 0:
        return images
    return torch.clamp(images + torch.randn_like(images) * std, -1.0, 1.0)


def augment_discriminator_input(images, config):
    return diff_augment(
        images,
        policy=config.get("diff_augment_policy", "none"),
        translation_ratio=config.get("diff_augment_translation_ratio", 0.125),
        cutout_ratio=config.get("diff_augment_cutout_ratio", 0.5),
    )


def init_dcgan_weights(module):
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        weight = getattr(module, "weight_orig", module.weight)
        nn.init.normal_(weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)


def generate_samples(generator, fixed_noise):
    was_training = generator.training
    generator.eval()
    with torch.no_grad():
        samples = generator(fixed_noise).detach()
    if was_training:
        generator.train()
    return samples


def get_evaluation_epochs(config, total_epochs, sample_interval):
    configured_epochs = config.get("evaluation_epochs")
    if configured_epochs:
        epochs = configured_epochs
    else:
        epochs = [epoch for epoch in DEFAULT_EPOCHS if epoch <= total_epochs]
        if total_epochs % sample_interval == 0:
            epochs.append(total_epochs)
    return sorted(set(epoch for epoch in epochs if epoch <= total_epochs))


def save_checkpoint(
    checkpoint_dir,
    epoch,
    generator,
    discriminator,
    g_optimizer,
    d_optimizer,
    epoch_row,
    config,
    dataloader_generator=None,
):
    epoch_number = epoch + 1
    generator_path = checkpoint_dir / f"generator_epoch_{epoch_number}.pth"
    discriminator_path = checkpoint_dir / f"discriminator_epoch_{epoch_number}.pth"
    state_path = checkpoint_dir / f"checkpoint_epoch_{epoch_number}.pt"

    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)
    torch.save(
        {
            "epoch": epoch,
            "epoch_number": epoch_number,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "g_optimizer_state_dict": g_optimizer.state_dict(),
            "d_optimizer_state_dict": d_optimizer.state_dict(),
            "epoch_metrics": epoch_row,
            "config": config,
            "rng_state": capture_rng_state(dataloader_generator),
        },
        state_path,
    )


def restore_training_checkpoint(
    checkpoint_path,
    generator,
    discriminator,
    g_optimizer,
    d_optimizer,
    device,
    current_config=None,
    dataloader_generator=None,
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_resume_config(checkpoint.get("config"), current_config)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
    d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
    restore_rng_state(checkpoint.get("rng_state"), dataloader_generator)
    return checkpoint["epoch"] + 1


def train(config, config_path, resume=None):
    seed = config.get("seed")
    if seed is not None:
        set_seed(seed)

    device = get_device(config.get("device", "auto"))
    IMAGE_SIZE = config["image_size"]
    Generator, Discriminator, model_file = get_model_classes(config)
    transform = get_transform(
        IMAGE_SIZE,
        config["augmentation"],
        config,
    )
    dataset_path = resolve_project_path(config["dataset_path"])
    dataset = My_dataset(dataset_path, transform=transform)   # 数据集位置
    expected_dataset_size = config.get("evaluation_real_count")
    if expected_dataset_size is not None and len(dataset) != expected_dataset_size:
        raise ValueError(
            f"Expected {expected_dataset_size} images, found {len(dataset)} in {dataset_path}."
        )
    batch_size, epochs = config["batch_size"], config["epochs"]
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset has {len(dataset)} images, which is smaller than batch_size={batch_size} "
            "while drop_last=True."
        )

    dataloader_generator = None
    if seed is not None:
        dataloader_generator = torch.Generator()
        dataloader_generator.manual_seed(seed)

    my_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config["num_workers"], # A100情况，其余改4
        pin_memory=device.type == "cuda",
        persistent_workers=config["num_workers"] > 0,
        generator=dataloader_generator
    )

    run_dir = create_run_dir(config["experiment_name"], config.get("run_root", "runs"))
    save_config_used(run_dir, config_path)

    discriminator = Discriminator().to(device)
    generator = Generator(noise_dim=config["noise_dim"]).to(device)
    generator.apply(init_dcgan_weights)
    discriminator.apply(init_dcgan_weights)

    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        betas=(config["beta1"], config["beta2"]),
        lr=config["d_lr"],
    )  # betas为adam算法两个动量参数
    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        betas=(config["beta1"], config["beta2"]),
        lr=config["g_lr"],
    )
    start_epoch = 0
    if resume is not None:
        start_epoch = restore_training_checkpoint(
            resume,
            generator,
            discriminator,
            g_optimizer,
            d_optimizer,
            device,
            current_config=config,
            dataloader_generator=dataloader_generator,
        )
        if start_epoch >= epochs:
            raise ValueError(
                f"Resume checkpoint starts at epoch {start_epoch}, but config epochs={epochs}."
            )
    criterion = nn.BCEWithLogitsLoss()
    real_label_value = config.get("real_label", 0.9)
    fake_label_value = config.get("fake_label", 0.1)
    g_steps = config.get("g_steps", 1)
    d_train_every = config.get("d_train_every", 1)
    checkpoint_interval = config.get("checkpoint_interval", 50)
    sample_interval = config.get("sample_interval", 10)
    sample_count = config.get("sample_count", 64)
    fixed_noise = create_fixed_noise(
        sample_count,
        config["noise_dim"],
        config.get("fixed_noise_seed", config.get("seed", 0)),
        device,
    )

    save_config(run_dir, config, len(dataset), dataset_path, config_path, model_file)
    save_environment(run_dir, device)
    save_source_code(run_dir, model_file)
    save_git_commit(run_dir)
    save_model_architecture(run_dir, generator, discriminator)
    parameter_counts = save_parameter_counts(run_dir, generator, discriminator)

    metrics_path = run_dir / "metrics.csv"
    epoch_metrics_path = run_dir / "epoch_metrics.csv"
    train_log_path = run_dir / "train.log"
    checkpoint_dir = run_dir / "checkpoints"

    epoch_metrics = []
    start_time = time.time()
    final_d_loss = None
    final_g_loss = None
    best_d_loss = None
    best_g_loss = None

    with open(train_log_path, "w", encoding="utf-8") as log_fp, \
            open(metrics_path, "w", newline="", encoding="utf-8") as metrics_fp, \
            open(epoch_metrics_path, "w", newline="", encoding="utf-8") as epoch_metrics_fp:
        metrics_writer = csv.DictWriter(
            metrics_fp,
            fieldnames=["epoch", "step", "d_loss", "g_loss", "D_real", "D_fake", "D_gap"]
        )
        epoch_metrics_writer = csv.DictWriter(
            epoch_metrics_fp,
            fieldnames=[
                "epoch", "avg_d_loss", "avg_g_loss", "avg_D_real", "avg_D_fake",
                "avg_D_gap", "epoch_seconds",
            ]
        )
        metrics_writer.writeheader()
        epoch_metrics_writer.writeheader()

        log_message(f"Using device: {device}", log_fp)

        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            epoch_rows = []
            instance_noise_std = get_instance_noise_std(config, epoch)

            for i, img in enumerate(my_dataloader):

                noise = torch.randn(batch_size, config["noise_dim"], 1, 1).to(device)
                real_img = img.to(device)
                fake_img = generator(noise).detach()

                real_label = torch.full(
                    (batch_size,),
                    real_label_value,
                    device=device,
                )
                fake_label = torch.full(
                    (batch_size,),
                    fake_label_value,
                    device=device,
                )
                discriminator_images = torch.cat([
                    add_instance_noise(real_img, instance_noise_std),
                    add_instance_noise(fake_img, instance_noise_std),
                ])
                discriminator_images = augment_discriminator_input(discriminator_images, config)
                real_input, fake_input = discriminator_images.split(batch_size)
                real_out = discriminator(real_input)
                fake_out = discriminator(fake_input)
                real_loss = criterion(real_out, real_label)
                fake_loss = criterion(fake_out, fake_label)

                d_loss = real_loss + fake_loss
                if i % d_train_every == 0:
                    d_optimizer.zero_grad(set_to_none=True)
                    d_loss.backward()
                    d_optimizer.step()
                    d_optimizer.zero_grad(set_to_none=True)

                g_loss = None
                set_requires_grad(discriminator, False)
                for _ in range(g_steps):
                    noise = torch.randn(batch_size, config["noise_dim"], 1, 1).to(device)
                    fake_img = generator(noise)
                    generator_input = augment_discriminator_input(
                        add_instance_noise(fake_img, instance_noise_std),
                        config,
                    )
                    output = discriminator(generator_input)

                    g_loss = criterion(output, real_label)
                    g_optimizer.zero_grad(set_to_none=True)

                    g_loss.backward()
                    g_optimizer.step()
                set_requires_grad(discriminator, True)

                d_loss_value = d_loss.detach().item()
                g_loss_value = g_loss.detach().item()
                d_real_value = torch.sigmoid(real_out.detach()).mean().item()
                d_fake_value = torch.sigmoid(fake_out.detach()).mean().item()
                d_gap_value = d_real_value - d_fake_value

                row = {
                    "epoch": epoch,
                    "step": i,
                    "d_loss": d_loss_value,
                    "g_loss": g_loss_value,
                    "D_real": d_real_value,
                    "D_fake": d_fake_value,
                    "D_gap": d_gap_value,
                }
                metrics_writer.writerow(row)
                metrics_fp.flush()
                epoch_rows.append(row)

                final_d_loss = d_loss_value
                final_g_loss = g_loss_value
                best_d_loss = d_loss_value if best_d_loss is None else min(best_d_loss, d_loss_value)
                best_g_loss = g_loss_value if best_g_loss is None else min(best_g_loss, g_loss_value)

                if (i + 1) % 5 == 0:
                    log_message('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                                'D_real: {:.6f},D_fake: {:.6f}'.format(
                        epoch, epochs, d_loss_value, g_loss_value,
                        d_real_value, d_fake_value
                    ), log_fp)
                if epoch == 0 and i == len(my_dataloader) - 1:          # 保存真实图像
                    save_img(img[:64, :, :, :], run_dir / "sample" / "real_images.png")
                if (epoch + 1) % sample_interval == 0 and i == len(my_dataloader) - 1:
                    samples = generate_samples(generator, fixed_noise)
                    save_img(samples, run_dir / "sample" / "fake_images_{}.png".format(epoch + 1))

            if epoch_rows:
                avg_d_loss = sum(row["d_loss"] for row in epoch_rows) / len(epoch_rows)
                avg_g_loss = sum(row["g_loss"] for row in epoch_rows) / len(epoch_rows)
                avg_D_real = sum(row["D_real"] for row in epoch_rows) / len(epoch_rows)
                avg_D_fake = sum(row["D_fake"] for row in epoch_rows) / len(epoch_rows)
                avg_D_gap = sum(row["D_gap"] for row in epoch_rows) / len(epoch_rows)
                epoch_row = {
                    "epoch": epoch,
                    "avg_d_loss": avg_d_loss,
                    "avg_g_loss": avg_g_loss,
                    "avg_D_real": avg_D_real,
                    "avg_D_fake": avg_D_fake,
                    "avg_D_gap": avg_D_gap,
                    "epoch_seconds": time.time() - epoch_start_time,
                }
                epoch_metrics_writer.writerow(epoch_row)
                epoch_metrics_fp.flush()
                epoch_metrics.append(epoch_row)

                if (epoch + 1) % checkpoint_interval == 0:
                    save_checkpoint(
                        checkpoint_dir,
                        epoch,
                        generator,
                        discriminator,
                        g_optimizer,
                        d_optimizer,
                        epoch_row,
                        config,
                        dataloader_generator,
                    )

    torch.save(generator.state_dict(), checkpoint_dir / "generator.pth")        # 保存权重文件
    torch.save(discriminator.state_dict(), checkpoint_dir / "discriminator.pth")
    torch.save(generator.state_dict(), checkpoint_dir / "generator_final.pth")
    torch.save(discriminator.state_dict(), checkpoint_dir / "discriminator_final.pth")

    plot_training_curves(run_dir, epoch_metrics)

    sample_metrics_path = None
    sample_metrics_rows = 0
    sample_metrics_missing = 0
    sample_metrics_error = None
    evaluation_epochs = get_evaluation_epochs(config, epochs, sample_interval)
    if evaluation_epochs:
        try:
            rows, missing, output_path = evaluate_run_samples(
                run_dir=run_dir,
                epochs=evaluation_epochs,
                output_path=run_dir / "sample_quality_metrics.csv",
            )
            sample_metrics_path = str(output_path)
            sample_metrics_rows = len(rows)
            sample_metrics_missing = len(missing)
            print(f"Sample quality metrics saved to {output_path}")
            if missing:
                print(f"Skipped {len(missing)} missing sample grids during evaluation.")
        except Exception as exc:
            sample_metrics_error = str(exc)
            print(f"Sample quality evaluation failed: {sample_metrics_error}")

    summary = {
        "experiment_name": config["experiment_name"],
        "augmentation": config["augmentation"],
        "config_file": str(config_path),
        "epochs": epochs,
        "dataset_size": len(dataset),
        "resolved_dataset_path": str(dataset_path),
        "model_variant": config.get("model_variant", f"legacy_{IMAGE_SIZE}"),
        "resume_checkpoint": str(Path(resume).resolve()) if resume else None,
        "parameter_counts": parameter_counts,
        "training_time_hours": (time.time() - start_time) / 3600,
        "mean_epoch_seconds": (
            sum(row["epoch_seconds"] for row in epoch_metrics) / len(epoch_metrics)
            if epoch_metrics else None
        ),
        "peak_accelerator_memory_mb": get_peak_memory_mb(device),
        "final_d_loss": final_d_loss,
        "final_g_loss": final_g_loss,
        "best_d_loss": best_d_loss,
        "best_g_loss": best_g_loss,
        "sample_quality_metrics": sample_metrics_path,
        "sample_quality_metrics_rows": sample_metrics_rows,
        "sample_quality_metrics_missing": sample_metrics_missing,
        "sample_quality_metrics_error": sample_metrics_error,
    }
    save_json(run_dir / "summary.json", summary)
    return run_dir


def main():
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    train(config, config_path, resume=args.resume)


if __name__ == "__main__":
    main()
