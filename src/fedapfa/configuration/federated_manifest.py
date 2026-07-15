"""Strict loading and expansion of the SHD FedAvg manifest."""

from __future__ import annotations

import copy
from pathlib import Path

from .federated_validation import paired_configuration_identity, validate_federated_config
from .loader import load_resolved_config
from .manifest import ManifestTask, _manifest_mapping

FEDERATED_SEEDS = (7, 17, 27)


def load_federated_config(path: str | Path) -> dict:
    config = load_resolved_config(path)
    validate_federated_config(config)
    return config


def load_federated_manifest(path: str | Path) -> list[ManifestTask]:
    manifest_path = Path(path).resolve()
    manifest = _manifest_mapping(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("collection") != "federated_baselines":
        raise ValueError("manifest must declare schema_version: 1 and collection: federated_baselines")
    seeds_name = manifest.get("seeds_file")
    if not isinstance(seeds_name, str) or not seeds_name:
        raise ValueError("manifest.seeds_file must be a path string")
    seeds = _manifest_mapping((manifest_path.parent / seeds_name).resolve()).get("seeds")
    if seeds != list(FEDERATED_SEEDS):
        raise ValueError(f"federated seeds must be exactly {list(FEDERATED_SEEDS)}")
    context = manifest.get("centralized_context")
    if not isinstance(context, str) or not context:
        raise ValueError("manifest.centralized_context must be a path string")

    experiments = manifest.get("experiments")
    if not isinstance(experiments, list) or len(experiments) != 2:
        raise ValueError("federated manifest must contain exactly two experiments")
    tasks: list[ManifestTask] = []
    templates = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    participations: set[float] = set()
    for entry in experiments:
        if not isinstance(entry, dict):
            raise ValueError("each federated manifest experiment must be a mapping")
        experiment = entry.get("id")
        config_name = entry.get("config")
        if not isinstance(experiment, str) or not experiment or experiment in seen_ids:
            raise ValueError(f"invalid or duplicate federated experiment id: {experiment!r}")
        if entry.get("mandatory") is not True:
            raise ValueError(f"federated experiment {experiment} must be mandatory")
        if not isinstance(config_name, str) or not config_name:
            raise ValueError(f"federated experiment {experiment} requires a config path")
        config_path = (manifest_path.parent / config_name).resolve()
        if config_path in seen_paths or manifest_path.parent not in config_path.parents:
            raise ValueError(f"invalid or duplicate federated config path: {config_name}")
        config = load_federated_config(config_path)
        if config["name"] != experiment or config.get("metadata", {}).get("experiment") != experiment:
            raise ValueError(f"federated experiment identity mismatch: {experiment}")
        templates.append(config)
        participations.add(float(config["federated"]["participation_fraction"]))
        seen_ids.add(experiment)
        seen_paths.add(config_path)
        for seed in seeds:
            resolved = copy.deepcopy(config)
            resolved["seed"] = seed
            validate_federated_config(resolved)
            tasks.append(
                ManifestTask(
                    experiment=experiment,
                    config_path=config_path,
                    seed=seed,
                    dataset="shd",
                    mode=resolved["mode"],
                    protocol=resolved["protocol"],
                    config=resolved,
                )
            )
    if participations != {0.25, 0.5}:
        raise ValueError("federated manifest must contain participation 0.25 and 0.50")
    if paired_configuration_identity(templates[0]) != paired_configuration_identity(templates[1]):
        raise ValueError("paired federated configurations differ outside participation settings")
    if len(tasks) != 6:
        raise ValueError("federated manifest must expand to exactly six tasks")
    return tasks
