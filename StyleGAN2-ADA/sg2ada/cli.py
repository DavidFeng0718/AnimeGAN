import argparse

from . import __version__
from .config import load_config
from .paths import DEFAULT_CONFIG_PATH
from .runtime import latest_snapshot, resolve_project_path


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Project StyleGAN2-ADA tools.")
    parser.add_argument("--version", action="version", version=f"sg2ada {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Run training from a JSON config.")
    train.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    train.add_argument("--dry-run", action="store_true")

    generate = subparsers.add_parser("generate", help="Generate samples from a network pickle.")
    generate.add_argument("args", nargs=argparse.REMAINDER)

    inspect = subparsers.add_parser("inspect", help="Print resolved config and latest snapshot.")
    inspect.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    if parsed.command == "train":
        from .train_entry import main as train_main

        train_args = ["--config", parsed.config]
        if parsed.dry_run:
            train_args.append("--dry-run")
        return train_main(train_args)
    if parsed.command == "generate":
        from .generate import main as generate_main

        return generate_main(parsed.args)
    if parsed.command == "inspect":
        config_path, config = load_config(parsed.config)
        run_dir = resolve_project_path(config["run_root"]) / config["experiment_name"]
        snapshot = latest_snapshot(config["run_root"], config["experiment_name"])
        print(f"config={config_path}")
        print(f"experiment={config['experiment_name']}")
        print(f"run_dir={run_dir}")
        print(f"latest_snapshot={snapshot if snapshot else 'none'}")
        return None
    raise RuntimeError(f"Unsupported command: {parsed.command}")
