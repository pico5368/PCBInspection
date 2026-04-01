"""Global configuration loading."""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load system configuration from YAML.

    Args:
        path: Config file path. Defaults to configs/default.yaml.

    Returns:
        Configuration dictionary.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
