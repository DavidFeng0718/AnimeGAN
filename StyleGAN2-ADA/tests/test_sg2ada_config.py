import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from sg2ada.config import load_config  # noqa: E402
from sg2ada.runtime import build_train_command, effective_training_config, latest_snapshot  # noqa: E402


def test_load_config_applies_new_defaults(tmp_path):
    source = PROJECT_DIR / "configs" / "smoke.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    data.pop("subset")
    data.pop("gamma")
    config_path = tmp_path / "smoke_legacy.json"
    config_path.write_text(json.dumps(data), encoding="utf-8")

    _, config = load_config(config_path)

    assert config["subset"] is None
    assert config["gamma"] is None


def test_build_train_command_exposes_subset_and_gamma():
    _, config = load_config(PROJECT_DIR / "configs" / "smoke.json")
    config["subset"] = 16
    config["gamma"] = 2.5
    config["resume"] = "latest-if-available"
    config = effective_training_config(config, "auto")

    _, command = build_train_command(config, PROJECT_DIR / "datasets" / "animegan_64_smoke.zip", True)

    assert "--subset" in command
    assert "16" in command
    assert "--gamma" in command
    assert "2.5" in command
    assert "--resume" in command
    assert "noresume" in command


def test_animefaces_512_config_loads():
    _, config = load_config(PROJECT_DIR / "configs" / "animefaces_512.json")

    assert config["dataset_name"] == "animefaces_danbooru_512"
    assert config["image_size"] == 512
    assert config["dataset_zip"] == "datasets/animefaces_512.zip"
    assert config["batch_size"] == 8


def test_latest_snapshot_picks_newest(tmp_path):
    run_root = tmp_path / "runs"
    first = run_root / "exp" / "00000-test" / "network-snapshot-000001.pkl"
    second = run_root / "exp" / "00001-test" / "network-snapshot-000002.pkl"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    assert latest_snapshot(run_root, "exp") == second
