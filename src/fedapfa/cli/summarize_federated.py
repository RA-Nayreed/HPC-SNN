"""Aggregate the mandatory three-seed SHD FedAvg reference executions."""

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

from fedapfa.configuration import FEDERATED_SEEDS, load_federated_manifest


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"expected a finite number, got {value!r}")
    return float(value)


def _statistics(values: list[float]) -> dict[str, float]:
    if len(values) != len(FEDERATED_SEEDS):
        raise ValueError("federated statistics require exactly three seed values")
    return {
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _scientific_identity(config: dict) -> dict:
    identity = copy.deepcopy(config)
    identity.pop("output_root", None)
    identity.pop("resume", None)
    identity.get("dataset", {}).pop("root", None)
    return identity


def _discover_runs(root: Path, expected_names: set[str]) -> tuple[dict, list[str]]:
    grouped: dict[tuple[str, int], list[tuple[Path, dict]]] = defaultdict(list)
    findings: list[str] = []
    if not root.is_dir():
        return grouped, [f"runs root does not exist: {root}"]
    for child in sorted(root.iterdir()):
        config_path = child / "resolved_config.yaml"
        if not child.is_dir() or not config_path.is_file():
            continue
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            findings.append(f"invalid resolved configuration {config_path}: {error}")
            continue
        if not isinstance(config, dict) or config.get("name") not in expected_names:
            continue
        seed = config.get("seed")
        if seed not in FEDERATED_SEEDS:
            findings.append(f"{config['name']} has unexpected seed {seed}")
            continue
        grouped[(config["name"], seed)].append((child, config))
    return grouped, findings


def _load_run(task, candidate: tuple[Path, dict]) -> tuple[dict | None, list[str]]:
    run_dir, config = candidate
    label = f"{task.experiment} seed {task.seed}"
    findings: list[str] = []
    if _scientific_identity(config) != _scientific_identity(task.config):
        findings.append(f"{label}: resolved configuration is incompatible with the manifest")
    try:
        acceptance = _read_json(run_dir / "acceptance.json")
        final = _read_json(run_dir / "final_metrics.json")
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        return None, findings + [f"{label}: missing or invalid final records ({error})"]
    if acceptance.get("completed") is not True or acceptance.get("accepted") is not True:
        findings.append(f"{label}: execution did not pass completion checks")
    if acceptance.get("scientific_status") != "not_claimed":
        findings.append(f"{label}: unexpected scientific status")
    if acceptance.get("protocol") != "independent_evaluation" or acceptance.get("seed") != task.seed:
        findings.append(f"{label}: acceptance protocol or seed is incompatible")
    if acceptance.get("reference_test_accuracy") is not None:
        findings.append(f"{label}: centralized context must not be configured as a reproduction target")
    try:
        spike_rates = {name: _finite(value) for name, value in final["mean_client_spike_rates"].items()}
        run = {
            "seed": task.seed,
            "run_directory": str(run_dir),
            "participation_fraction": _finite(config["federated"]["participation_fraction"]),
            "best_validation_accuracy": _finite(final["best_validation_accuracy"]),
            "official_test_accuracy": _finite(final["test"]["accuracy"]),
            "selected_round": _finite(final["selected_round"]),
            "final_validation_accuracy": _finite(final["final_validation_accuracy"]),
            "logical_communication_bytes": _finite(final["logical_communication"]["cumulative_total_bytes"]),
            "execution_time_seconds": _finite(final["execution_time_seconds"]),
            "mean_client_update_l2_norm": _finite(final["mean_client_update_l2_norm"]),
            "mean_client_spike_rates": spike_rates,
            "parameter_count": int(final["parameter_count"]),
            "split_id": acceptance["split_id"],
            "partition_id": acceptance["partition_id"],
            "model_initialization_id": acceptance["model_initialization_id"],
            "git_commit": acceptance["git_commit"],
        }
    except (KeyError, TypeError, ValueError) as error:
        return None, findings + [f"{label}: final metrics are incomplete or non-finite ({error})"]
    if not spike_rates or run["parameter_count"] <= 0:
        findings.append(f"{label}: spike rates or parameter count are invalid")
    return run, findings


def _centralized_context(manifest_path: Path) -> tuple[dict, list[str]]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    context_path = (manifest_path.parent / manifest["centralized_context"]).resolve()
    findings: list[str] = []
    try:
        summary = _read_json(context_path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        return {}, [f"centralized context is missing or invalid: {error}"]
    if summary.get("valid") is not True:
        findings.append("centralized context is not valid")
    if summary.get("required_seeds") != list(FEDERATED_SEEDS):
        findings.append("centralized context does not use seeds 7, 17, and 27")
    matches = [
        experiment
        for experiment in summary.get("experiments", [])
        if experiment.get("experiment") == "shd_lif_independent_evaluation"
        and experiment.get("protocol") == "independent_evaluation"
    ]
    if len(matches) != 1 or matches[0].get("completed") is not True:
        findings.append("completed centralized SHD LIF independent-evaluation context is unavailable")
        return {}, findings
    experiment = matches[0]
    if experiment.get("required_seeds") != list(FEDERATED_SEEDS) or sorted(
        run.get("seed") for run in experiment.get("runs", [])
    ) != list(FEDERATED_SEEDS):
        findings.append("centralized SHD LIF context has incompatible seed records")
    return {
        "source": str(context_path),
        "experiment": experiment["experiment"],
        "protocol": experiment["protocol"],
        "official_test_accuracy": experiment["metrics"]["official_test_accuracy"],
        "runs": [
            {"seed": run["seed"], "official_test_accuracy": run["official_test_accuracy"]}
            for run in experiment["runs"]
        ],
    }, findings


def summarize_federated(manifest: str | Path, runs_root: str | Path, output_dir: str | Path) -> dict:
    manifest_path = Path(manifest).resolve()
    tasks = load_federated_manifest(manifest_path)
    root = Path(runs_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    order = list(dict.fromkeys(task.experiment for task in tasks))
    grouped, findings = _discover_runs(root, {task.config["name"] for task in tasks})
    valid_runs: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        candidates = grouped.get((task.config["name"], task.seed), [])
        label = f"{task.experiment} seed {task.seed}"
        if not candidates:
            findings.append(f"{label}: missing mandatory execution")
            continue
        if len(candidates) != 1:
            findings.append(f"{label}: duplicated mandatory execution ({len(candidates)} directories)")
            continue
        run, run_findings = _load_run(task, candidates[0])
        findings.extend(run_findings)
        if run is not None and not run_findings:
            valid_runs[task.experiment].append(run)

    context, context_findings = _centralized_context(manifest_path)
    findings.extend(context_findings)
    experiments = []
    metric_names = (
        "best_validation_accuracy",
        "official_test_accuracy",
        "selected_round",
        "final_validation_accuracy",
        "logical_communication_bytes",
        "execution_time_seconds",
        "mean_client_update_l2_norm",
    )
    for experiment_name in order:
        template = next(task for task in tasks if task.experiment == experiment_name)
        runs = sorted(valid_runs.get(experiment_name, []), key=lambda value: value["seed"])
        result = {
            "experiment": experiment_name,
            "protocol": "independent_evaluation",
            "participation_fraction": template.config["federated"]["participation_fraction"],
            "required_seeds": list(FEDERATED_SEEDS),
            "completed_seed_count": len(runs),
            "completed": len(runs) == len(FEDERATED_SEEDS),
            "scientific_status": "not_claimed",
            "runs": runs,
            "metrics": {},
            "mean_client_spike_rates": {},
        }
        if result["completed"]:
            for metric_name in metric_names:
                result["metrics"][metric_name] = _statistics([run[metric_name] for run in runs])
            layer_names = sorted({name for run in runs for name in run["mean_client_spike_rates"]})
            for name in layer_names:
                if any(name not in run["mean_client_spike_rates"] for run in runs):
                    findings.append(f"{experiment_name}: client spike layer {name} is missing for a seed")
                else:
                    result["mean_client_spike_rates"][name] = _statistics(
                        [run["mean_client_spike_rates"][name] for run in runs]
                    )
            parameter_counts = {run["parameter_count"] for run in runs}
            if len(parameter_counts) != 1:
                findings.append(f"{experiment_name}: parameter counts differ across seeds")
            result["parameter_count"] = min(parameter_counts)
            central_mean = context.get("official_test_accuracy", {}).get("mean")
            result["centralized_minus_federated_accuracy"] = (
                None if central_mean is None else central_mean - result["metrics"]["official_test_accuracy"]["mean"]
            )
        else:
            result["parameter_count"] = None
            result["centralized_minus_federated_accuracy"] = None
        experiments.append(result)

    by_participation = {experiment["participation_fraction"]: experiment for experiment in experiments}
    paired = []
    if all(by_participation.get(value, {}).get("completed") for value in (0.25, 0.5)):
        low = {run["seed"]: run for run in by_participation[0.25]["runs"]}
        high = {run["seed"]: run for run in by_participation[0.5]["runs"]}
        for seed in FEDERATED_SEEDS:
            for identity in ("split_id", "partition_id", "model_initialization_id"):
                if low[seed][identity] != high[seed][identity]:
                    findings.append(f"seed {seed}: paired participation executions differ in {identity}")
            paired.append(
                {
                    "seed": seed,
                    "official_test_accuracy_difference": high[seed]["official_test_accuracy"]
                    - low[seed]["official_test_accuracy"],
                    "best_validation_accuracy_difference": high[seed]["best_validation_accuracy"]
                    - low[seed]["best_validation_accuracy"],
                    "selected_round_difference": high[seed]["selected_round"] - low[seed]["selected_round"],
                    "logical_communication_bytes_difference": high[seed]["logical_communication_bytes"]
                    - low[seed]["logical_communication_bytes"],
                    "execution_time_seconds_difference": high[seed]["execution_time_seconds"]
                    - low[seed]["execution_time_seconds"],
                }
            )
    else:
        findings.append("paired participation differences require all six completed executions")
    paired_summary = {
        "definition": "participation_0_50_minus_participation_0_25_by_seed",
        "runs": paired,
        "official_test_accuracy_difference": (
            _statistics([record["official_test_accuracy_difference"] for record in paired]) if len(paired) == 3 else {}
        ),
    }
    summary = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "runs_root": str(root),
        "expected_task_count": 6,
        "required_seeds": list(FEDERATED_SEEDS),
        "valid": not findings,
        "validation_findings": findings,
        "centralized_context": context,
        "experiments": experiments,
        "paired_participation_differences": paired_summary,
    }
    _write_json(output / "federated_summary.json", summary)
    _write_csv(output / "federated_summary.csv", experiments)
    _write_markdown(output / "federated_summary.md", summary)
    return summary


def _write_json(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, experiments: list[dict]) -> None:
    rows = []
    for experiment in experiments:
        row = {
            "experiment": experiment["experiment"],
            "protocol": experiment["protocol"],
            "participation_fraction": experiment["participation_fraction"],
            "completed": experiment["completed"],
            "completed_seed_count": experiment["completed_seed_count"],
            "scientific_status": experiment["scientific_status"],
            "parameter_count": experiment["parameter_count"],
            "centralized_minus_federated_accuracy": experiment["centralized_minus_federated_accuracy"],
        }
        for metric, values in experiment["metrics"].items():
            for statistic_name, value in values.items():
                row[f"{metric}_{statistic_name}"] = value
        for layer, values in experiment["mean_client_spike_rates"].items():
            for statistic_name, value in values.items():
                row[f"{layer}_client_spike_rate_{statistic_name}"] = value
        rows.append(row)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _formatted(values: dict) -> str:
    return "n/a" if not values else f"{values['mean']:.6g} ± {values['sample_standard_deviation']:.6g}"


def _write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Federated SHD LIF evaluation summary",
        "",
        f"Status: **{'valid' if summary['valid'] else 'invalid'}**",
        "",
        "The centralized SHD LIF independent evaluation is contextual evidence, not an acceptance threshold.",
        "",
        "| Experiment | Participation | Seeds | Best validation accuracy | Official test accuracy | "
        "Communication bytes | Scientific status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for experiment in summary["experiments"]:
        metrics = experiment["metrics"]
        lines.append(
            f"| {experiment['experiment']} | {experiment['participation_fraction']:.2f} | "
            f"{experiment['completed_seed_count']}/3 | "
            f"{_formatted(metrics.get('best_validation_accuracy', {}))} | "
            f"{_formatted(metrics.get('official_test_accuracy', {}))} | "
            f"{_formatted(metrics.get('logical_communication_bytes', {}))} | "
            f"{experiment['scientific_status']} |"
        )
    if summary["validation_findings"]:
        lines.extend(["", "## Validation findings", ""])
        lines.extend(f"- {finding}" for finding in summary["validation_findings"])
    lines.extend(
        [
            "",
            "Logical communication is model-tensor accounting and is not measured network traffic.",
            "A not_claimed status indicates that no verified reproduction target is configured.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize mandatory SHD FedAvg scientific evaluations.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = summarize_federated(args.manifest, args.runs_root, args.output_dir)
    if not summary["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
