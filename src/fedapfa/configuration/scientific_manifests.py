"""Strict manifests for heterogeneity and published-protocol evaluations."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

from .federated_manifest import FEDERATED_SEEDS, load_federated_config
from .federated_validation import validate_federated_config
from .manifest import ManifestTask, _manifest_mapping


@dataclass(frozen=True)
class ContextTask:
    seed: int
    experiment: str
    source_record: dict


def _load_tasks(path: str | Path, collection: str, experiment_count: int) -> list[ManifestTask]:
    manifest_path = Path(path).resolve()
    manifest = _manifest_mapping(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("collection") != collection:
        raise ValueError(f"manifest must declare schema_version: 1 and collection: {collection}")
    seeds_name = manifest.get("seeds_file")
    if not isinstance(seeds_name, str) or not seeds_name:
        raise ValueError("manifest.seeds_file must be a path string")
    seeds = _manifest_mapping((manifest_path.parent / seeds_name).resolve()).get("seeds")
    if seeds != list(FEDERATED_SEEDS):
        raise ValueError(f"scientific seeds must be exactly {list(FEDERATED_SEEDS)}")
    entries = manifest.get("experiments")
    if not isinstance(entries, list) or len(entries) != experiment_count:
        raise ValueError(f"{collection} must contain exactly {experiment_count} experiments")
    tasks: list[ManifestTask] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("mandatory") is not True:
            raise ValueError("every scientific experiment must be a mandatory mapping")
        experiment = entry.get("id")
        config_name = entry.get("config")
        if not isinstance(experiment, str) or not experiment or experiment in seen_ids:
            raise ValueError(f"invalid or duplicate experiment id: {experiment!r}")
        if not isinstance(config_name, str) or not config_name:
            raise ValueError(f"experiment {experiment} requires a config path")
        config_path = (manifest_path.parent / config_name).resolve()
        if config_path in seen_paths or manifest_path.parent not in config_path.parents:
            raise ValueError(f"invalid or duplicate experiment config: {config_name}")
        template = load_federated_config(config_path)
        if template["name"] != experiment or template.get("metadata", {}).get("experiment") != experiment:
            raise ValueError(f"experiment identity mismatch: {experiment}")
        for seed in seeds:
            config = copy.deepcopy(template)
            config["seed"] = seed
            validate_federated_config(config)
            tasks.append(
                ManifestTask(
                    experiment=experiment,
                    config_path=config_path,
                    seed=seed,
                    dataset=config["dataset"]["name"],
                    mode=config["mode"],
                    protocol=config["protocol"],
                    config=config,
                )
            )
        seen_ids.add(experiment)
        seen_paths.add(config_path)
    return tasks


def _fixed_training_identity(config: dict) -> dict:
    federation = copy.deepcopy(config["federated"])
    federation.pop("partition")
    return {
        "dataset": config["dataset"],
        "model": config["model"],
        "training": config["training"],
        "subset": config["subset"],
        "federated": federation,
        "seed_streams": config["seed_streams"],
    }


def load_heterogeneity_manifest(path: str | Path) -> list[ManifestTask]:
    tasks = _load_tasks(path, "heterogeneity_evaluation", 3)
    if len(tasks) != 9:
        raise ValueError("heterogeneity manifest must expand to exactly nine tasks")
    templates = [tasks[index].config for index in range(0, len(tasks), 3)]
    if len({_fixed_training_identity(config).__repr__() for config in templates}) != 1:
        raise ValueError("heterogeneity configurations differ outside the partition")
    treatments = {
        (config["federated"]["partition"]["method"], config["federated"]["partition"].get("alpha"))
        for config in templates
    }
    if treatments != {("stratified_iid", None), ("label_dirichlet", 1.0), ("label_dirichlet", 0.1)}:
        raise ValueError("heterogeneity manifest has an incompatible treatment matrix")
    manifest_path = Path(path).resolve()
    context = _manifest_mapping(manifest_path).get("contextual_evidence")
    if not isinstance(context, dict) or context.get("new_training_task") is not False:
        raise ValueError("heterogeneity manifest must declare contextual evidence without training")
    summary = (manifest_path.parent / context.get("summary", "")).resolve()
    if not summary.is_file():
        raise ValueError(f"contextual federated summary is unavailable: {summary}")
    if context.get("alpha") != 0.5 or context.get("participation_fraction") != 0.5:
        raise ValueError("contextual evidence must identify alpha 0.5 and participation 0.50")
    return tasks


def load_published_fedsnn_manifest(path: str | Path) -> list[ManifestTask]:
    tasks = _load_tasks(path, "published_fedsnn", 2)
    if len(tasks) != 6 or {task.dataset for task in tasks} != {"cifar10"}:
        raise ValueError("published Fed-SNN manifest must expand to exactly six CIFAR-10 tasks")
    templates = [tasks[index].config for index in range(0, len(tasks), 3)]
    if {config["protocol"] for config in templates} != {"paper_reported_evaluation"}:
        raise ValueError("published Fed-SNN manifest has an incompatible protocol matrix")
    if {config["name"] for config in templates} != {
        "cifar10_fedsnn_paper_reported_iid_evaluation",
        "cifar10_fedsnn_paper_reported_noniid_evaluation",
    }:
        raise ValueError("published Fed-SNN manifest has incompatible Table I experiment identities")
    if len({_fixed_training_identity(config).__repr__() for config in templates}) != 1:
        raise ValueError("published Fed-SNN configurations differ outside distribution-specific fields")
    treatments = {
        (
            config["name"],
            config["federated"]["local_epochs"],
            config["federated"]["partition"]["method"],
            config["federated"]["partition"].get("alpha"),
            config["acceptance"]["descriptive_reference_accuracy"],
        )
        for config in templates
    }
    if treatments != {
        ("cifar10_fedsnn_paper_reported_iid_evaluation", 5, "fedsnn_random_iid", None, 0.7644),
        (
            "cifar10_fedsnn_paper_reported_noniid_evaluation",
            5,
            "fedsnn_balanced_label_dirichlet",
            0.5,
            0.7394,
        ),
    }:
        raise ValueError("published Fed-SNN manifest has an incompatible treatment matrix")
    return tasks


def load_heterogeneity_context_tasks(manifest_path: str | Path) -> list[ContextTask]:
    path = Path(manifest_path).resolve()
    manifest = _manifest_mapping(path)
    context = manifest.get("contextual_evidence")
    if not isinstance(context, dict):
        raise ValueError("contextual_evidence must be a mapping")
    summary_path = (path.parent / context.get("summary", "")).resolve()
    summary = _manifest_mapping(summary_path)
    matching = [
        value for value in summary.get("experiments", []) if value.get("experiment") == context.get("experiment")
    ]
    if len(matching) != 1 or matching[0].get("completed") is not True:
        raise ValueError("exactly one completed contextual experiment is required")
    runs = matching[0].get("runs")
    if not isinstance(runs, list) or sorted(value.get("seed") for value in runs) != list(FEDERATED_SEEDS):
        raise ValueError("contextual executions must contain seeds 7, 17, and 27 exactly once")
    required = {
        "git_commit",
        "model_initialization_id",
        "partition_id",
        "run_directory",
        "split_id",
    }
    if any(required.difference(value) for value in runs):
        raise ValueError("contextual execution identities are incomplete")
    return [ContextTask(int(value["seed"]), str(context["experiment"]), dict(value)) for value in runs]
