from pathlib import Path
from typing import Any

import yaml

from .validation import validate_config


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate one resolved experiment configuration."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    validate_config(config)
    return config
