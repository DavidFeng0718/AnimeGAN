import argparse

from .config import load_config
from .paths import DEFAULT_CONFIG_PATH
from .runtime import (
    build_train_command,
    effective_training_config,
    prepare_dataset,
    resolve_device,
    run_command,
    save_experiment_snapshot,
)
from .paths import UPSTREAM_DIR


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run StyleGAN2-ADA through a project JSON config."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Experiment JSON config (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print upstream StyleGAN2-ADA training options.",
    )
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    config_path, config = load_config(parsed.config)
    dataset = prepare_dataset(config)
    device = resolve_device(config, parsed.dry_run)
    config = effective_training_config(config, device)
    outdir, command = build_train_command(config, dataset, parsed.dry_run)
    save_experiment_snapshot(outdir, config_path, config, dataset, parsed.dry_run)
    run_command(command, UPSTREAM_DIR, config)
