from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return config


def resolve_project_path(path_value: str | Path, root_dir: Path = ROOT_DIR) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return root_dir / path


def get_nested(config: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
