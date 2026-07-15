"""Aggregate the mandatory three-seed centralized results."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from fedapfa.configuration import load_centralized_manifest
from fedapfa.configuration.manifest import CENTRALIZED_SEEDS


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _scientific_identity(config: dict) -> dict:
    identity = copy.deepcopy(config)
    identity.pop("output_root", None)
    identity.pop("resume", None)
    identity.get("dataset", {}).pop("root", None)
    return identity


def _finite_number(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"expected a finite number, got {value!r}")
    return float(value)


def _stats(values: list[float]) -> dict[str, float]:
    if len(values) != len(CENTRALIZED_SEEDS):
        raise ValueError(f"statistics require {len(CENTRALIZED_SEEDS)} seed values")
    return {
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _discover_runs(runs_root: Path, expected_names: set[str]):
    grouped = defaultdict(list)
    errors = []
    if not runs_root.is_dir():
        return grouped, [f"runs root does not exist: {runs_root}"]
    unexpected_seeds = []
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        config_path = child / "resolved_config.yaml"
        if not config_path.is_file():
            continue
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            errors.append(f"invalid resolved configuration: {config_path}")
            continue
        if not isinstance(config, dict) or config.get("name") not in expected_names:
            continue
        seed = config.get("seed")
        if seed not in CENTRALIZED_SEEDS:
            unexpected_seeds.append(f"{config['name']} seed {seed}")
            continue
        grouped[(config["name"], seed)].append((child, config))
    if unexpected_seeds:
        errors.append("unexpected mandatory-experiment seeds: " + ", ".join(unexpected_seeds))
    return grouped, errors


def _load_run(task, run_dir: Path, actual_config: dict) -> tuple[dict | None, list[str]]:
    errors = []
    label = f"{task.experiment} seed {task.seed}"
    if _scientific_identity(actual_config) != _scientific_identity(task.config):
        errors.append(f"{label}: resolved configuration is incompatible with the manifest")
    if actual_config.get("protocol") != task.protocol:
        errors.append(f"{label}: protocol mismatch would mix independent-evaluation and published-protocol results")
    try:
        acceptance = _read_json(run_dir / "acceptance.json")
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        return None, errors + [f"{label}: missing or invalid acceptance.json ({error})"]
    try:
        final = _read_json(run_dir / "final_metrics.json")
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        return None, errors + [f"{label}: missing or invalid final_metrics.json ({error})"]

    if acceptance.get("completed") is not True:
        failures = acceptance.get("completion_failures", [])
        errors.append(f"{label}: run is incomplete or failed: {failures}")
    if acceptance.get("protocol") != task.protocol:
        errors.append(f"{label}: acceptance protocol mismatch")
    if acceptance.get("seed") != task.seed:
        errors.append(f"{label}: acceptance seed mismatch")
    if acceptance.get("mode") != "scientific_evaluation":
        errors.append(f"{label}: acceptance mode is not scientific_evaluation")
    expected_acceptance = task.config["acceptance"]
    if acceptance.get("reference_test_accuracy") != expected_acceptance["reference_test_accuracy"]:
        errors.append(f"{label}: reference accuracy differs from the manifest")
    if acceptance.get("tolerance") != expected_acceptance["absolute_tolerance"]:
        errors.append(f"{label}: tolerance differs from the manifest")

    try:
        test = final["test"]
        spike_rates = test["spike_rates"]
        run = {
            "seed": task.seed,
            "run_directory": str(run_dir),
            "best_validation_accuracy": _finite_number(final["best_selection_accuracy"]),
            "official_test_accuracy": _finite_number(test["accuracy"]),
            "runtime_seconds": _finite_number(final["runtime_seconds"]),
            "peak_cuda_memory_bytes": _finite_number(final["peak_cuda_memory_bytes"]),
            "layer_spike_rates": {name: _finite_number(value) for name, value in spike_rates.items()},
            "parameter_count": int(final["parameter_count"]),
            "scientific_status": acceptance["scientific_status"],
            "absolute_accuracy_difference": acceptance.get("absolute_accuracy_difference"),
            "git_commit": acceptance.get("git_commit"),
        }
    except (KeyError, TypeError, ValueError) as error:
        return None, errors + [f"{label}: final metrics are incomplete or non-finite ({error})"]
    if run["parameter_count"] <= 0:
        errors.append(f"{label}: parameter count must be positive")
    if not run["layer_spike_rates"]:
        errors.append(f"{label}: layer spike rates are missing")
    difference = run["absolute_accuracy_difference"]
    if difference is not None:
        try:
            run["absolute_accuracy_difference"] = _finite_number(difference)
        except ValueError as error:
            errors.append(f"{label}: invalid reproduction delta ({error})")
    reference = expected_acceptance["reference_test_accuracy"]
    if reference is None:
        if run["scientific_status"] != "not_claimed" or difference is not None:
            errors.append(f"{label}: null reference must have not_claimed status and null delta")
    else:
        calculated = abs(run["official_test_accuracy"] - reference)
        if difference is None or not math.isclose(difference, calculated, rel_tol=0.0, abs_tol=1e-12):
            errors.append(f"{label}: reproduction delta does not match official test accuracy")
        expected_status = "passed" if calculated <= expected_acceptance["absolute_tolerance"] else "failed"
        if run["scientific_status"] != expected_status:
            errors.append(f"{label}: invalid scientific status {run['scientific_status']!r}")
    return run, errors


def summarize_centralized(manifest: str | Path, runs_root: str | Path, output_dir: str | Path) -> dict:
    tasks = load_centralized_manifest(manifest)
    root = Path(runs_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    experiment_order = list(dict.fromkeys(task.experiment for task in tasks))
    task_by_key = {(task.config["name"], task.seed): task for task in tasks}
    expected_names = {task.config["name"] for task in tasks}
    grouped, errors = _discover_runs(root, expected_names)
    valid_runs = defaultdict(list)

    for key, task in task_by_key.items():
        candidates = grouped.get(key, [])
        label = f"{task.experiment} seed {task.seed}"
        if not candidates:
            errors.append(f"{label}: missing mandatory run")
            continue
        if len(candidates) > 1:
            errors.append(f"{label}: duplicated mandatory run ({len(candidates)} directories)")
            continue
        run, run_errors = _load_run(task, *candidates[0])
        errors.extend(run_errors)
        if run is not None and not run_errors:
            valid_runs[task.experiment].append(run)

    experiments = []
    for experiment in experiment_order:
        experiment_tasks = [task for task in tasks if task.experiment == experiment]
        template = experiment_tasks[0]
        runs = sorted(valid_runs.get(experiment, []), key=lambda item: item["seed"])
        result = {
            "experiment": experiment,
            "config_name": template.config["name"],
            "dataset": template.dataset,
            "model_name": template.config["model"]["name"],
            "attention_variant": template.config["model"]["attention"]["variant"],
            "protocol": template.protocol,
            "required_seeds": list(CENTRALIZED_SEEDS),
            "completed_seed_count": len(runs),
            "completed": len(runs) == len(CENTRALIZED_SEEDS),
            "reference_test_accuracy": template.config["acceptance"]["reference_test_accuracy"],
            "tolerance": template.config["acceptance"]["absolute_tolerance"],
            "runs": runs,
        }
        if len(runs) == len(CENTRALIZED_SEEDS):
            parameter_counts = {run["parameter_count"] for run in runs}
            if len(parameter_counts) != 1:
                errors.append(f"{experiment}: parameter counts differ across seeds")
            result["parameter_count"] = min(parameter_counts)
            result["metrics"] = {
                "best_validation_accuracy": _stats([run["best_validation_accuracy"] for run in runs]),
                "official_test_accuracy": _stats([run["official_test_accuracy"] for run in runs]),
                "runtime_seconds": _stats([run["runtime_seconds"] for run in runs]),
                "peak_cuda_memory_bytes": _stats([run["peak_cuda_memory_bytes"] for run in runs]),
            }
            layer_names = sorted({name for run in runs for name in run["layer_spike_rates"]})
            layer_statistics = {}
            for name in layer_names:
                if any(name not in run["layer_spike_rates"] for run in runs):
                    errors.append(f"{experiment}: spike-rate layer {name} is missing for one or more seeds")
                    continue
                layer_statistics[name] = _stats([run["layer_spike_rates"][name] for run in runs])
            result["layer_spike_rates"] = layer_statistics
            reference = result["reference_test_accuracy"]
            if reference is None:
                result["scientific_status"] = "not_claimed"
                result["reproduction_delta"] = None
            else:
                deltas = [run["absolute_accuracy_difference"] for run in runs]
                if any(delta is None for delta in deltas):
                    errors.append(f"{experiment}: configured reference is missing a reproduction delta")
                    result["reproduction_delta"] = None
                else:
                    result["reproduction_delta"] = _stats(deltas)
                result["scientific_status"] = (
                    "passed" if all(run["scientific_status"] == "passed" for run in runs) else "failed"
                )
        else:
            result["parameter_count"] = None
            result["metrics"] = {}
            result["layer_spike_rates"] = {}
            result["scientific_status"] = "not_claimed" if result["reference_test_accuracy"] is None else "failed"
            result["reproduction_delta"] = None
        experiments.append(result)

    summary = {
        "schema_version": 1,
        "manifest": str(Path(manifest)),
        "runs_root": str(root),
        "expected_task_count": 18,
        "required_seeds": list(CENTRALIZED_SEEDS),
        "valid": not errors,
        "errors": errors,
        "experiments": experiments,
    }
    _write_json(output / "centralized_summary.json", summary)
    _write_csv(output / "centralized_summary.csv", experiments)
    _write_markdown(output / "centralized_summary.md", summary)
    return summary


def _write_json(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, experiments: list[dict]) -> None:
    rows = []
    for experiment in experiments:
        row = {
            "experiment": experiment["experiment"],
            "config_name": experiment["config_name"],
            "dataset": experiment["dataset"],
            "model_name": experiment["model_name"],
            "attention_variant": experiment["attention_variant"],
            "protocol": experiment["protocol"],
            "completed": experiment["completed"],
            "completed_seed_count": experiment["completed_seed_count"],
            "scientific_status": experiment["scientific_status"],
            "reference_test_accuracy": experiment["reference_test_accuracy"],
            "tolerance": experiment["tolerance"],
            "parameter_count": experiment["parameter_count"],
        }
        for metric, values in experiment["metrics"].items():
            for statistic, value in values.items():
                row[f"{metric}_{statistic}"] = value
        for layer, values in experiment["layer_spike_rates"].items():
            for statistic, value in values.items():
                row[f"{layer}_spike_rate_{statistic}"] = value
        delta = experiment["reproduction_delta"]
        if delta:
            for statistic, value in delta.items():
                row[f"reproduction_delta_{statistic}"] = value
        rows.append(row)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_statistic(values: dict) -> str:
    if not values:
        return "n/a"
    return f"{values['mean']:.6g} ± {values['sample_standard_deviation']:.6g}"


def _write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Centralized evaluation summary",
        "",
        f"Status: **{'valid' if summary['valid'] else 'invalid'}**",
        "",
        "Independent-evaluation and published-protocol results are reported as separate "
        "experiment rows and are never pooled.",
        "",
        "| Experiment | Protocol | Seeds | Best validation accuracy | Official test accuracy | Scientific status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for experiment in summary["experiments"]:
        metrics = experiment["metrics"]
        lines.append(
            f"| {experiment['experiment']} | {experiment['protocol']} | "
            f"{experiment['completed_seed_count']}/3 | "
            f"{_format_statistic(metrics.get('best_validation_accuracy', {}))} | "
            f"{_format_statistic(metrics.get('official_test_accuracy', {}))} | "
            f"{experiment['scientific_status']} |"
        )
    if summary["errors"]:
        lines.extend(["", "## Validation errors", ""])
        lines.extend(f"- {error}" for error in summary["errors"])
    lines.extend(
        [
            "",
            "A not_claimed status means the literature target is null or unverified; it is not a reproduction pass.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize mandatory centralized scientific evaluations.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = summarize_centralized(args.manifest, args.runs_root, args.output_dir)
    if not summary["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
