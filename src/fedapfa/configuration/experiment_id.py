"""Stable, collision-resistant experiment identities."""

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def canonical_config(config: Mapping[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def experiment_id(config: Mapping[str, Any]) -> str:
    identity = json.loads(canonical_config(config))
    identity.pop("resume", None)
    identity.pop("output_root", None)
    digest = hashlib.sha256(canonical_config(identity).encode()).hexdigest()[:8]
    return f"{config['name']}-seed{config['seed']}-{digest}"


def expand_sweep(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if config.get("mode") != "sweep":
        return [dict(config)]
    runs = []
    for value in config["sweep"]["values"]:
        resolved = json.loads(canonical_config(config))
        resolved["mode"] = "smoke"
        resolved["name"] = f"{config['name']}-lambda-{value:g}"
        resolved["model"]["attention"]["lambda"] = value
        resolved.pop("sweep")
        runs.append(resolved)
    return runs
