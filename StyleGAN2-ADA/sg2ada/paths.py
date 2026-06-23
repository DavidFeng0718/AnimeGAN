from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = PROJECT_DIR / "upstream"
DEFAULT_CONFIG_PATH = "configs/baseline.json"


def resolve_project_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()
