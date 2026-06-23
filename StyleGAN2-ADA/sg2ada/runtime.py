import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime

from .paths import PROJECT_DIR, UPSTREAM_DIR, resolve_project_path


def bool_arg(value):
    return "1" if value else "0"


def metrics_arg(metrics):
    if metrics is None:
        return "none"
    if isinstance(metrics, str):
        return metrics
    if isinstance(metrics, list):
        return ",".join(metrics) if metrics else "none"
    raise ValueError("Config 'metrics' must be a string, list, or null.")


def command_env(config=None):
    env = os.environ.copy()
    if config is not None and config.get("device") in {"auto", "mps"}:
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    current_pythonpath = env.get("PYTHONPATH")
    paths = [str(UPSTREAM_DIR)]
    if current_pythonpath:
        paths.append(current_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def run_command(command, cwd, config=None):
    print(" ".join(str(part) for part in command))
    subprocess.run(command, cwd=str(cwd), env=command_env(config), check=True)


def prepare_dataset(config):
    dataset_path = resolve_project_path(config["dataset_path"])
    dataset_zip = resolve_project_path(config["dataset_zip"])

    if dataset_zip.exists() and not config["force_prepare_dataset"]:
        return dataset_zip

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if not dataset_path.is_dir() and dataset_path.suffix.lower() != ".zip":
        raise ValueError(f"Dataset path must be a directory or .zip archive: {dataset_path}")

    if not config["prepare_dataset"]:
        return dataset_path

    if dataset_zip.exists() and config["force_prepare_dataset"]:
        dataset_zip.unlink()

    dataset_zip.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "dataset_tool.py",
        "--source",
        str(dataset_path),
        "--dest",
        str(dataset_zip),
        "--width",
        str(config["image_size"]),
        "--height",
        str(config["image_size"]),
        "--resize-filter",
        config["resize_filter"],
    ]
    if config["dataset_transform"] is not None:
        command.extend(["--transform", config["dataset_transform"]])
    if config.get("max_images") is not None:
        command.extend(["--max-images", str(config["max_images"])])

    run_command(command, UPSTREAM_DIR, config)
    return dataset_zip


def resolve_device(config, dry_run=False):
    requested = config["device"]
    if dry_run and requested == "auto":
        return "auto"

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to launch StyleGAN2-ADA training.") from exc

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested device=cuda, but CUDA is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Config requested device=mps, but Apple MPS is not available.")
    return requested


def effective_training_config(config, device):
    effective = dict(config)
    effective["resolved_device"] = device
    if device in {"mps", "cpu"}:
        effective["gpus"] = 1
        effective["fp32"] = True
        effective["nhwc"] = False
        effective["allow_tf32"] = False
        effective["nobench"] = True
    return effective


def latest_snapshot(run_root, experiment_name):
    run_dir = resolve_project_path(run_root) / experiment_name
    snapshots = sorted(
        run_dir.glob("*/network-snapshot-*.pkl"),
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    return snapshots[-1] if snapshots else None


def resolve_resume(config):
    resume = config["resume"]
    if resume not in {"latest", "latest-if-available"}:
        return resume

    snapshot = latest_snapshot(config["run_root"], config["experiment_name"])
    if snapshot is not None:
        return str(snapshot)
    if resume == "latest-if-available":
        return "noresume"
    raise FileNotFoundError(
        "resume=latest was requested, but no network-snapshot-*.pkl was found under "
        f"{resolve_project_path(config['run_root']) / config['experiment_name']}"
    )


def build_train_command(config, dataset, dry_run=False):
    outdir = resolve_project_path(config["run_root"]) / config["experiment_name"]
    outdir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "train.py",
        "--outdir",
        str(outdir),
        "--data",
        str(dataset),
        "--device",
        config["resolved_device"],
        "--gpus",
        str(config["gpus"]),
        "--cfg",
        config["cfg"],
        "--kimg",
        str(config["kimg"]),
        "--batch",
        str(config["batch_size"]),
        "--snap",
        str(config["snap"]),
        "--metrics",
        metrics_arg(config["metrics"]),
        "--seed",
        str(config["seed"]),
        "--cond",
        bool_arg(config["cond"]),
        "--mirror",
        bool_arg(config["mirror"]),
        "--aug",
        config["aug"],
        "--resume",
        resolve_resume(config),
        "--freezed",
        str(config["freezed"]),
        "--fp32",
        bool_arg(config["fp32"]),
        "--nhwc",
        bool_arg(config["nhwc"]),
        "--allow-tf32",
        bool_arg(config["allow_tf32"]),
        "--nobench",
        bool_arg(config["nobench"]),
        "--workers",
        str(config["workers"]),
    ]
    if config.get("subset") is not None:
        command.extend(["--subset", str(config["subset"])])
    if config.get("gamma") is not None:
        command.extend(["--gamma", str(config["gamma"])])
    if config["aug"] != "noaug":
        command.extend(["--augpipe", config["augpipe"]])
    if config["aug"] == "ada" and config.get("ada_target") is not None:
        command.extend(["--target", str(config["ada_target"])])
    if config["aug"] == "fixed":
        command.extend(["--p", str(config["fixed_aug_p"])])
    if dry_run:
        command.append("--dry-run")
    return outdir, command


def save_json(path, data):
    with path.open("w", encoding="utf-8") as fp:
        import json

        json.dump(data, fp, indent=4)
        fp.write("\n")


def save_experiment_snapshot(outdir, config_path, config, dataset, dry_run=False):
    snapshot_dir = outdir / "_wrapper"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, snapshot_dir / "config_used.json")

    config_to_save = dict(config)
    config_to_save.update(
        {
            "resolved_dataset": str(dataset),
            "resolved_config": str(config_path),
            "dry_run": dry_run,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )
    save_json(snapshot_dir / "config.json", config_to_save)

    environment = [
        f"Python {platform.python_version()}",
        f"Executable {sys.executable}",
        f"OS {platform.platform()}",
        f"Upstream {UPSTREAM_DIR}",
    ]
    with (snapshot_dir / "environment.txt").open("w", encoding="utf-8") as fp:
        fp.write("\n".join(environment) + "\n")

    try:
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=str(PROJECT_DIR),
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception as exc:
        status = f"Unavailable: {exc}\n"
    with (snapshot_dir / "git_status.txt").open("w", encoding="utf-8") as fp:
        fp.write(status)
