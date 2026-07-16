"""Validate and summarize the four-treatment SHD heterogeneity evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from fedapfa.configuration import FEDERATED_SEEDS, load_heterogeneity_manifest
from fedapfa.metrics.client_fairness import PROXY_EXPLANATION

TREATMENTS = (
    ("IID", "shd_lif_iid_participation_0_50"),
    ("Dirichlet alpha 1.0", "shd_lif_dirichlet_alpha_1_0_participation_0_50"),
    ("Dirichlet alpha 0.5 contextual evidence", "shd_lif_dirichlet_alpha_0_5_participation_0_50"),
    ("Dirichlet alpha 0.1", "shd_lif_dirichlet_alpha_0_1_participation_0_50"),
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _stats(values: list[float]) -> dict[str, float]:
    if len(values) != 3 or any(not math.isfinite(value) for value in values):
        raise ValueError("treatment statistics require three finite seed values")
    return {"mean": statistics.mean(values), "sample_standard_deviation": statistics.stdev(values)}


def trapezoidal_area(curve: list[list[float]]) -> float:
    """Integrate validation accuracy over communication-round number."""

    if len(curve) < 2:
        raise ValueError("convergence area requires at least two points")
    points = [(_finite(x, "round number"), _finite(y, "validation accuracy")) for x, y in curve]
    if any(right[0] <= left[0] for left, right in zip(points, points[1:], strict=False)):
        raise ValueError("convergence round numbers must increase strictly")
    return sum(
        (right_x - left_x) * (left_y + right_y) / 2
        for (left_x, left_y), (right_x, right_y) in zip(points, points[1:], strict=False)
    )


def _discover_new(root: Path, names: set[str]) -> dict[tuple[str, int], list[Path]]:
    grouped: dict[tuple[str, int], list[Path]] = defaultdict(list)
    if not root.is_dir():
        return grouped
    for child in sorted(root.iterdir()):
        config_path = child / "resolved_config.yaml"
        if not child.is_dir() or not config_path.is_file():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if config.get("name") in names and config.get("seed") in FEDERATED_SEEDS:
            grouped[(config["name"], int(config["seed"]))].append(child)
    return grouped


def _protocol_settings(config: dict) -> dict:
    return {
        "dataset": {key: value for key, value in config["dataset"].items() if key != "root"},
        "model": config["model"],
        "federated": {
            key: config["federated"].get(key)
            for key in (
                "clients",
                "clients_per_round",
                "participation_fraction",
                "rounds",
                "local_epochs",
                "local_batch_size",
                "optimizer",
                "learning_rate",
                "weight_decay",
                "gradient_clip",
                "client_sampling",
            )
        },
    }


def _new_run(task, path: Path) -> dict:
    config = yaml.safe_load((path / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
    acceptance = _read_json(path / "acceptance.json")
    final = _read_json(path / "final_metrics.json")
    partition = _read_json(path / "partition.json")
    rounds = _read_jsonl(path / "round_metrics.jsonl")
    if acceptance.get("accepted") is not True or acceptance.get("completed") is not True:
        raise ValueError("execution is not accepted and completed")
    if acceptance.get("seed") != task.seed or acceptance.get("protocol") != "independent_evaluation":
        raise ValueError("acceptance seed or protocol is incompatible")
    if len(rounds) != 100 or [value.get("round_number") for value in rounds] != list(range(1, 101)):
        raise ValueError("communication-round records are incomplete or duplicated")
    curve = [[value["round_number"], value["validation_accuracy"]] for value in rounds]
    return {
        "seed": task.seed,
        "run_directory": str(path),
        "contextual": False,
        "split_id": acceptance["split_id"],
        "partition_id": acceptance["partition_id"],
        "model_initialization_id": acceptance["model_initialization_id"],
        "git_commit": acceptance["git_commit"],
        "protocol_settings": _protocol_settings(config),
        "best_validation_accuracy": _finite(final["best_validation_accuracy"], "best validation accuracy"),
        "official_test_accuracy": _finite(final["test"]["accuracy"], "official test accuracy"),
        "validation_macro_f1": _finite(final["selected_validation"]["macro_f1"], "validation macro-F1"),
        "selected_round": _finite(final["selected_round"], "selected round"),
        "final_validation_accuracy": _finite(final["final_validation_accuracy"], "final validation accuracy"),
        "validation_to_test_difference": _finite(final["best_validation_accuracy"], "validation accuracy")
        - _finite(final["test"]["accuracy"], "test accuracy"),
        "convergence_area": trapezoidal_area(curve),
        "validation_curve": curve,
        "fairness_proxy": final["client_distribution_weighted_validation_accuracy"],
        "partition_diagnostics": partition["diagnostic_statistics"],
        "mean_client_update_l2_norm": _finite(final["mean_client_update_l2_norm"], "update norm"),
        "mean_update_alignment": _finite(final["mean_client_update_cosine_similarity"], "update alignment"),
        "mean_client_spike_rates": final["mean_client_spike_rates"],
        "execution_time_seconds": _finite(final["execution_time_seconds"], "execution duration"),
        "logical_communication_bytes": _finite(
            final["logical_communication"]["cumulative_total_bytes"], "logical communication"
        ),
        "peak_cuda_memory_bytes": final["peak_cuda_memory_bytes"],
        "official_test_reevaluated": False,
    }


def _context_runs(root: Path, summary_path: Path) -> list[dict]:
    paths = sorted(root.rglob("context.json")) if root.is_dir() else []
    if len(paths) != 3:
        raise ValueError(f"context root must contain exactly three records, found {len(paths)}")
    summary = _read_json(summary_path)
    matches = [
        value
        for value in summary.get("experiments", [])
        if value.get("experiment") == "shd_lif_dirichlet_alpha_0_5_participation_0_50"
    ]
    if len(matches) != 1:
        raise ValueError("committed federated summary has no unique contextual treatment")
    sources = {int(value["seed"]): value for value in matches[0].get("runs", [])}
    values = []
    for path in paths:
        record = _read_json(path)
        seed = int(record.get("seed"))
        source = sources.get(seed)
        identities = record.get("source_identities", {})
        if source is None or any(
            identities.get(key) != source.get(key)
            for key in ("split_id", "partition_id", "model_initialization_id", "git_commit")
        ):
            raise ValueError(f"context seed {seed} does not match the committed source identities")
        if record.get("official_test_reevaluated") is not False:
            raise ValueError(f"context seed {seed} indicates an official-test re-evaluation")
        historical = record["historical_metrics"]
        curve = historical["validation_curve"]
        values.append(
            {
                "seed": seed,
                "run_directory": record["source_run_directory"],
                "context_record": str(path),
                "contextual": True,
                "split_id": identities["split_id"],
                "partition_id": identities["partition_id"],
                "model_initialization_id": identities["model_initialization_id"],
                "git_commit": identities["git_commit"],
                "protocol_settings": record["protocol_settings"],
                "best_validation_accuracy": _finite(historical["best_validation_accuracy"], "best accuracy"),
                "official_test_accuracy": _finite(historical["official_test_accuracy"], "test accuracy"),
                "validation_macro_f1": _finite(record["validation"]["macro_f1"], "macro-F1"),
                "selected_round": _finite(historical["selected_round"], "selected round"),
                "final_validation_accuracy": _finite(historical["final_validation_accuracy"], "final accuracy"),
                "validation_to_test_difference": _finite(historical["best_validation_accuracy"], "validation")
                - _finite(historical["official_test_accuracy"], "test"),
                "convergence_area": trapezoidal_area(curve),
                "validation_curve": curve,
                "fairness_proxy": record["client_distribution_weighted_validation_accuracy"],
                "partition_diagnostics": record["partition_diagnostics"]["aggregate_statistics"],
                "mean_client_update_l2_norm": _finite(historical["mean_client_update_l2_norm"], "update norm"),
                "mean_update_alignment": None,
                "mean_client_spike_rates": historical["mean_client_spike_rates"],
                "execution_time_seconds": _finite(historical["execution_time_seconds"], "duration"),
                "logical_communication_bytes": _finite(historical["logical_communication_bytes"], "communication"),
                "peak_cuda_memory_bytes": None,
                "official_test_reevaluated": False,
            }
        )
    if sorted(value["seed"] for value in values) != list(FEDERATED_SEEDS):
        raise ValueError("context records must contain seeds 7, 17, and 27 exactly once")
    return sorted(values, key=lambda value: value["seed"])


def _summarize_treatment(label: str, experiment: str, runs: list[dict]) -> dict:
    metrics = {}
    for key in (
        "best_validation_accuracy",
        "official_test_accuracy",
        "validation_macro_f1",
        "selected_round",
        "convergence_area",
        "final_validation_accuracy",
        "validation_to_test_difference",
        "mean_client_update_l2_norm",
        "execution_time_seconds",
        "logical_communication_bytes",
    ):
        metrics[key] = _stats([run[key] for run in runs])
    alignments = [run["mean_update_alignment"] for run in runs if run["mean_update_alignment"] is not None]
    metrics["mean_update_alignment"] = (
        _stats(alignments)
        if len(alignments) == 3
        else {
            "available": False,
            "reason": "Historical local update tensors were not retained.",
        }
    )
    layers = sorted({name for run in runs for name in run["mean_client_spike_rates"]})
    return {
        "label": label,
        "experiment": experiment,
        "contextual_evidence": bool(runs[0]["contextual"]),
        "seeds": [run["seed"] for run in runs],
        "runs": runs,
        "metrics": metrics,
        "fairness_proxy_statistics": {
            key: _stats([run["fairness_proxy"]["statistics"][key] for run in runs])
            for key in ("minimum", "10th_percentile", "median", "mean", "maximum", "population_standard_deviation")
        },
        "partition_statistics": {
            measure: {
                statistic_name: _stats([run["partition_diagnostics"][measure][statistic_name] for run in runs])
                for statistic_name in ("minimum", "maximum", "mean", "median", "population_standard_deviation")
            }
            for measure in (
                "label_entropy_bits",
                "normalized_label_entropy",
                "missing_labels",
                "jensen_shannon_divergence_bits",
            )
        },
        "mean_client_spike_rates": {
            name: _stats([_finite(run["mean_client_spike_rates"][name], f"{name} spike rate") for run in runs])
            for name in layers
        },
    }


def _plots(output: Path, treatments: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    labels = [value["label"] for value in treatments]

    def bars(path: str, title: str, values: list[float], ylabel: str) -> None:
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.bar(range(4), values)
        axis.set_xticks(range(4), labels, rotation=20, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        figure.tight_layout()
        figure.savefig(output / path, dpi=160)
        plt.close(figure)

    bars(
        "accuracy_by_partition.png",
        "Official-test accuracy by partition",
        [value["metrics"]["official_test_accuracy"]["mean"] for value in treatments],
        "accuracy",
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    for treatment in treatments:
        curves = treatment["runs"]
        x = [point[0] for point in curves[0]["validation_curve"]]
        mean_y = [statistics.mean(run["validation_curve"][index][1] for run in curves) for index in range(len(x))]
        axis.plot(x, mean_y, label=treatment["label"])
    axis.set_xlabel("communication-round number")
    axis.set_ylabel("validation accuracy")
    axis.set_title("Validation convergence")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "convergence_by_partition.png", dpi=160)
    plt.close(figure)
    bars(
        "label_distribution_by_partition.png",
        "Client-to-training label divergence",
        [value["partition_statistics"]["jensen_shannon_divergence_bits"]["mean"]["mean"] for value in treatments],
        "Jensen-Shannon divergence (bits)",
    )
    bars(
        "update_alignment_by_partition.png",
        "Client update alignment",
        [
            0.0 if value["contextual_evidence"] else value["metrics"]["mean_update_alignment"]["mean"]
            for value in treatments
        ],
        "cosine similarity (context unavailable)",
    )
    first_layer = next(iter(treatments[0]["mean_client_spike_rates"]))
    bars(
        "spike_rate_by_partition.png",
        f"Mean client spike rate ({first_layer})",
        [value["mean_client_spike_rates"][first_layer]["mean"] for value in treatments],
        "spike rate",
    )


def summarize_heterogeneity(
    manifest: str | Path,
    runs_root: str | Path,
    context_root: str | Path,
    federated_summary: str | Path,
    output_dir: str | Path,
) -> dict:
    tasks = load_heterogeneity_manifest(manifest)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    findings: list[str] = []
    grouped = _discover_new(Path(runs_root), {task.experiment for task in tasks})
    runs_by_experiment: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        candidates = grouped.get((task.experiment, task.seed), [])
        if len(candidates) != 1:
            findings.append(f"{task.experiment} seed {task.seed}: expected one execution, found {len(candidates)}")
            continue
        try:
            runs_by_experiment[task.experiment].append(_new_run(task, candidates[0]))
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            findings.append(f"{task.experiment} seed {task.seed}: {error}")
    try:
        runs_by_experiment[TREATMENTS[2][1]] = _context_runs(Path(context_root), Path(federated_summary))
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        findings.append(f"contextual evidence: {error}")
    for seed in FEDERATED_SEEDS:
        seed_runs = [
            next((run for run in runs_by_experiment.get(experiment, []) if run["seed"] == seed), None)
            for _, experiment in TREATMENTS
        ]
        if any(run is None for run in seed_runs):
            continue
        reference = seed_runs[0]
        if any(run["split_id"] != reference["split_id"] for run in seed_runs[1:]):
            findings.append(f"seed {seed}: SHD split identity differs across treatments")
        if any(run["model_initialization_id"] != reference["model_initialization_id"] for run in seed_runs[1:]):
            findings.append(f"seed {seed}: model initialization identity differs across treatments")
        if any(run["protocol_settings"] != reference["protocol_settings"] for run in seed_runs[1:]):
            findings.append(f"seed {seed}: architecture or fixed training settings differ across treatments")
        if len({run["partition_id"] for run in seed_runs}) != 4:
            findings.append(f"seed {seed}: treatment partition identities are not distinct")
    treatments = []
    for label, experiment in TREATMENTS:
        runs = sorted(runs_by_experiment.get(experiment, []), key=lambda value: value["seed"])
        if [run["seed"] for run in runs] == list(FEDERATED_SEEDS):
            treatments.append(_summarize_treatment(label, experiment, runs))
    summary = {
        "schema_version": 1,
        "valid": not findings and len(treatments) == 4,
        "validation_findings": findings,
        "training_task_count": 9,
        "context_task_count": 3,
        "required_seeds": list(FEDERATED_SEEDS),
        "treatment_order": [label for label, _ in TREATMENTS],
        "convergence_area_definition": "Trapezoidal area with communication-round number on the x-axis.",
        "client_fairness_proxy_definition": PROXY_EXPLANATION,
        "treatments": treatments,
    }
    (output / "heterogeneity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_csv(output / "heterogeneity_summary.csv", treatments)
    _write_markdown(output / "heterogeneity_summary.md", summary)
    if summary["valid"]:
        _plots(output, treatments)
    return summary


def _write_csv(path: Path, treatments: list[dict]) -> None:
    rows = []
    for treatment in treatments:
        row = {"treatment": treatment["label"], "seeds": ";".join(map(str, treatment["seeds"]))}
        for metric, values in treatment["metrics"].items():
            if "mean" in values:
                row[f"{metric}_mean"] = values["mean"]
                row[f"{metric}_sample_standard_deviation"] = values["sample_standard_deviation"]
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(metrics: dict, key: str) -> str:
    mean = metrics[key]["mean"]
    deviation = metrics[key]["sample_standard_deviation"]
    return f"{mean:.6g} ± {deviation:.6g}"


def _write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# SHD heterogeneity evaluation",
        "",
        f"Status: **{'valid' if summary['valid'] else 'invalid'}**",
        "",
        summary["client_fairness_proxy_definition"],
        "",
        summary["convergence_area_definition"],
        "",
        "| Treatment | Seeds | Best validation accuracy | Official-test accuracy | Validation macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for value in summary["treatments"]:
        metrics = value["metrics"]
        lines.append(
            f"| {value['label']} | 3 | {_format_metric(metrics, 'best_validation_accuracy')} | "
            f"{_format_metric(metrics, 'official_test_accuracy')} | {_format_metric(metrics, 'validation_macro_f1')} |"
        )
    if summary["validation_findings"]:
        lines.extend(["", "## Validation findings", ""])
        lines.extend(f"- {value}" for value in summary["validation_findings"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the SHD heterogeneity evaluation.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--context-root", required=True)
    parser.add_argument("--federated-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = summarize_heterogeneity(
        args.manifest, args.runs_root, args.context_root, args.federated_summary, args.output_dir
    )
    if not summary["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
