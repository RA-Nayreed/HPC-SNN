"""Small explicit YAML composition loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .validation import validate_config


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _load(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        raise ValueError(f"configuration include cycle: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    includes = data.pop("include", [])
    if not isinstance(includes, list):
        raise ValueError("include must be a list")
    resolved = {}
    for included in includes:
        included_path = (path.parent / included).resolve()
        resolved = _merge(resolved, _load(included_path, stack + (path,)))
    return _merge(resolved, data)


def load_config(path: str | Path) -> dict[str, Any]:
    config = _load(Path(path), ())
    validate_config(config)
    return config
