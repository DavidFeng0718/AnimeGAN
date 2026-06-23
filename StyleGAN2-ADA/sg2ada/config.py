import json
import os
from pathlib import Path

from .paths import DEFAULT_CONFIG_PATH, PROJECT_DIR


REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "experiment_name",
    "dataset_path",
    "dataset_zip",
    "prepare_dataset",
    "force_prepare_dataset",
    "image_size",
    "dataset_transform",
    "resize_filter",
    "run_root",
    "device",
    "gpus",
    "cfg",
    "kimg",
    "batch_size",
    "snap",
    "metrics",
    "seed",
    "cond",
    "mirror",
    "aug",
    "augpipe",
    "resume",
    "freezed",
    "fp32",
    "nhwc",
    "allow_tf32",
    "nobench",
    "workers",
}

VALID_CFGS = {"auto", "stylegan2", "paper256", "paper512", "paper1024", "cifar"}
VALID_AUGS = {"noaug", "ada", "fixed"}
VALID_AUGPIPES = {
    "blit",
    "geom",
    "color",
    "filter",
    "noise",
    "cutout",
    "bg",
    "bgc",
    "bgcf",
    "bgcfn",
    "bgcfnc",
}
VALID_TRANSFORMS = {None, "center-crop", "center-crop-wide"}
VALID_RESIZE_FILTERS = {"box", "lanczos"}
VALID_DEVICES = {"auto", "cuda", "mps", "cpu"}
RESUME_ALIASES = {"latest", "latest-if-available"}


def load_config(config_path=DEFAULT_CONFIG_PATH):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        cwd_path = config_path.resolve()
        config_path = cwd_path if cwd_path.exists() else (PROJECT_DIR / config_path).resolve()
    with config_path.open("r", encoding="utf-8") as fp:
        config = json.load(fp)
    apply_config_defaults(config)
    apply_env_overrides(config)
    validate_config(config)
    return config_path, config


def apply_config_defaults(config):
    config.setdefault("subset", None)
    config.setdefault("gamma", None)
    config.setdefault("run_tags", [])


def apply_env_overrides(config):
    env_map = {
        "STYLEGAN_DEVICE": "device",
        "STYLEGAN_DATASET_PATH": "dataset_path",
        "DATASET_DIR": "dataset_path",
        "STYLEGAN_DATASET_ZIP": "dataset_zip",
        "STYLEGAN_RESUME": "resume",
    }
    for env_key, config_key in env_map.items():
        value = os.environ.get(env_key)
        if value:
            config[config_key] = value


def validate_config(config):
    missing = sorted(REQUIRED_CONFIG_KEYS - config.keys())
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")

    if config["schema_version"] != 1:
        raise ValueError("StyleGAN2-ADA configs must use schema_version=1.")

    if not isinstance(config["experiment_name"], str) or not config["experiment_name"]:
        raise ValueError("Config 'experiment_name' must be a non-empty string.")

    for key in ["image_size", "gpus", "kimg", "batch_size", "snap", "seed", "workers"]:
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"Config '{key}' must be a positive integer.")

    if config["image_size"] & (config["image_size"] - 1) != 0:
        raise ValueError("Config 'image_size' must be a power of two.")

    if config["gpus"] & (config["gpus"] - 1) != 0:
        raise ValueError("Config 'gpus' must be a power of two.")

    if config["batch_size"] % config["gpus"] != 0:
        raise ValueError("Config 'batch_size' must be divisible by 'gpus'.")

    if config["cfg"] not in VALID_CFGS:
        raise ValueError(f"Unsupported cfg: {config['cfg']}")

    if config["aug"] not in VALID_AUGS:
        raise ValueError(f"Unsupported aug: {config['aug']}")

    if config["augpipe"] not in VALID_AUGPIPES:
        raise ValueError(f"Unsupported augpipe: {config['augpipe']}")

    if config["aug"] == "fixed":
        p = config.get("fixed_aug_p")
        if not isinstance(p, (int, float)) or not 0.0 <= p <= 1.0:
            raise ValueError("Config 'fixed_aug_p' must be in [0, 1] when aug=fixed.")

    if config["aug"] == "ada":
        target = config.get("ada_target")
        if target is not None and (
            not isinstance(target, (int, float)) or not 0.0 <= target <= 1.0
        ):
            raise ValueError("Config 'ada_target' must be in [0, 1].")

    if config["dataset_transform"] not in VALID_TRANSFORMS:
        raise ValueError(f"Unsupported dataset_transform: {config['dataset_transform']}")

    if config["resize_filter"] not in VALID_RESIZE_FILTERS:
        raise ValueError(f"Unsupported resize_filter: {config['resize_filter']}")

    validate_optional_positive_int(config, "max_images")
    validate_optional_positive_int(config, "subset")

    if config["gamma"] is not None and (
        not isinstance(config["gamma"], (int, float)) or config["gamma"] < 0
    ):
        raise ValueError("Config 'gamma' must be null or a non-negative number.")

    if config["device"] not in VALID_DEVICES:
        raise ValueError(f"Unsupported device: {config['device']}")

    if config["device"] in {"mps", "cpu"} and config["gpus"] != 1:
        raise ValueError("MPS/CPU training only supports gpus=1.")

    if not isinstance(config["resume"], str):
        raise ValueError("Config 'resume' must be a string.")

    for key in [
        "prepare_dataset",
        "force_prepare_dataset",
        "cond",
        "mirror",
        "fp32",
        "nhwc",
        "allow_tf32",
        "nobench",
    ]:
        if not isinstance(config[key], bool):
            raise ValueError(f"Config '{key}' must be boolean.")


def validate_optional_positive_int(config, key):
    if config.get(key) is not None:
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"Config '{key}' must be null or a positive integer.")
