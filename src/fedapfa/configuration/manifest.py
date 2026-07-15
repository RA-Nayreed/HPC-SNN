"""Strict expansion of the mandatory centralized experiment manifest."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml

from .loader import load_config
from .validation import validate_config

CENTRALIZED_SEEDS = (7, 17, 27)
FORBIDDEN_SCIENTIFIC_LABELS = ("reduced-sample", "reduced_sample", "sweep", "memorization_validation")


@dataclass(frozen=True)
class ManifestTask:
    experiment: str
    config_path: Path
    seed: int
    dataset: str
    mode: str
    protocol: str
    config: dict


def _manifest_mapping(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("centralized manifest root must be a mapping")
    return data


def load_centralized_manifest(path: str | Path) -> list[ManifestTask]:
    manifest_path = Path(path).resolve()
    manifest = _manifest_mapping(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("collection") != "centralized":
        raise ValueError("manifest must declare schema_version: 1 and collection: centralized")

    seeds_name = manifest.get("seeds_file")
    if not isinstance(seeds_name, str) or not seeds_name:
        raise ValueError("manifest.seeds_file must be a path string")
    seeds_path = (manifest_path.parent / seeds_name).resolve()
    seeds_data = _manifest_mapping(seeds_path)
    seeds = seeds_data.get("seeds")
    if seeds != list(CENTRALIZED_SEEDS):
        raise ValueError(f"centralized seeds must be exactly {list(CENTRALIZED_SEEDS)}")

    experiments = manifest.get("experiments")
    if not isinstance(experiments, list) or len(experiments) != 6:
        raise ValueError("centralized manifest must contain exactly six experiments")
    tasks = []
    seen_ids = set()
    seen_paths = set()
    for entry in experiments:
        if not isinstance(entry, dict):
            raise ValueError("each manifest experiment must be a mapping")
        experiment = entry.get("id")
        config_name = entry.get("config")
        if not isinstance(experiment, str) or not experiment:
            raise ValueError("manifest experiment id must be a non-empty string")
        if experiment in seen_ids:
            raise ValueError(f"duplicate manifest experiment id: {experiment}")
        if entry.get("mandatory") is not True:
            raise ValueError(f"centralized experiment {experiment} must be mandatory")
        if not isinstance(config_name, str) or not config_name:
            raise ValueError(f"manifest experiment {experiment} requires a config path")
        lowered = f"{experiment} {config_name}".lower()
        if any(label in lowered for label in FORBIDDEN_SCIENTIFIC_LABELS):
            raise ValueError(f"forbidden reduced-sample label in manifest entry: {experiment}")
        config_path = (manifest_path.parent / config_name).resolve()
        if config_path in seen_paths:
            raise ValueError(f"duplicate manifest config: {config_name}")
        if manifest_path.parent not in config_path.parents:
            raise ValueError("centralized configs must live below the manifest directory")
        config = load_config(config_path)
        if config["mode"] != "scientific_evaluation":
            raise ValueError(f"manifest experiment {experiment} is not mode: scientific_evaluation")
        seen_ids.add(experiment)
        seen_paths.add(config_path)
        for seed in seeds:
            resolved = copy.deepcopy(config)
            resolved["seed"] = seed
            validate_config(resolved)
            tasks.append(
                ManifestTask(
                    experiment=experiment,
                    config_path=config_path,
                    seed=seed,
                    dataset=resolved["dataset"]["name"],
                    mode=resolved["mode"],
                    protocol=resolved["protocol"],
                    config=resolved,
                )
            )
    if len(tasks) != 18:
        raise ValueError("centralized manifest must expand to exactly 18 tasks")
    return tasks
