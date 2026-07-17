"""Summarize the two Fed-SNN Table I treatments without an acceptance threshold."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

from fedapfa.configuration import FEDERATED_SEEDS, load_published_fedsnn_manifest

PAPER_TREATMENTS = {
    "cifar10_fedsnn_paper_reported_iid_evaluation": {
        "distribution": "iid",
        "alpha": None,
        "descriptive_reference_accuracy": 0.7644,
    },
    "cifar10_fedsnn_paper_reported_noniid_evaluation": {
        "distribution": "label_dirichlet_non_iid",
        "alpha": 0.5,
        "descriptive_reference_accuracy": 0.7394,
    },
}
SCIENTIFIC_STATUS = "equivalence_not_established"


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite(value, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _stats(values: list[float]) -> dict[str, float]:
    if len(values) != 3:
        raise ValueError("Fed-SNN Table I statistics require three seeds")
    return {
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
    }


def _validated_run(task, path: Path) -> dict:
    treatment = PAPER_TREATMENTS[task.experiment]
    acceptance = _read_json(path / "acceptance.json")
    final = _read_json(path / "final_metrics.json")
    rounds = [
        json.loads(line)
        for line in (path / "round_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if acceptance.get("completed") is not True or acceptance.get("accepted") is not True:
        raise ValueError("execution is not accepted and completed")
    if acceptance.get("scientific_status") != SCIENTIFIC_STATUS:
        raise ValueError(f"scientific status must be {SCIENTIFIC_STATUS}")
    if acceptance.get("reference_test_accuracy") is not None or acceptance.get("tolerance") is not None:
        raise ValueError("paper accuracy must not be configured as an acceptance target")
    if acceptance.get("descriptive_reference_accuracy") != treatment["descriptive_reference_accuracy"]:
        raise ValueError("descriptive paper reference is incompatible")
    if len(rounds) != 100 or [record.get("round_number") for record in rounds] != list(range(1, 101)):
        raise ValueError("communication-round records are incomplete")
    if final.get("selected_round") != 100 or final.get("checkpoint_selection") != "final_round":
        raise ValueError("the durable final-round checkpoint was not selected")
    if any(
        final.get(key) is not None
        for key in (
            "best_validation_accuracy",
            "final_validation_accuracy",
            "selected_validation",
            "client_distribution_weighted_validation_accuracy",
        )
    ):
        raise ValueError("run reports unavailable internal-validation metrics")
    data_protocol = final["data_protocol"]
    expected_counts = {
        "examples_available_before_validation_separation": 50000,
        "examples_used_for_client_training": 50000,
        "examples_used_for_validation": 0,
        "official_test_examples": 10000,
        "official_test_access_count": 1,
    }
    if any(data_protocol.get(key) != value for key, value in expected_counts.items()):
        raise ValueError("training, validation, or official-test counts are incompatible")
    official_accuracy = _finite(final["test"]["accuracy"], "official-test accuracy")
    reference = float(treatment["descriptive_reference_accuracy"])
    signed_difference = (official_accuracy - reference) * 100
    config = task.config["federated"]
    record = {
        "experiment": task.experiment,
        "protocol": task.protocol,
        "distribution": treatment["distribution"],
        "alpha": treatment["alpha"],
        "seed": task.seed,
        "run_directory": str(path),
        "completed": True,
        "final_round": 100,
        "official_test_accuracy": official_accuracy,
        "official_test_macro_f1": _finite(final["test"]["macro_f1"], "official-test macro-F1"),
        "descriptive_paper_reference": reference,
        "signed_difference_percentage_points": signed_difference,
        "absolute_difference_percentage_points": abs(signed_difference),
        "local_epochs": int(config["local_epochs"]),
        "total_clients": int(config["clients"]),
        "participating_clients": int(config["clients_per_round"]),
        "timesteps": int(task.config["model"]["timesteps"]),
        "momentum": _finite(config["momentum"], "momentum"),
        "weight_decay": _finite(config["weight_decay"], "weight decay"),
        "aggregation_weighting": config["aggregation_weighting"],
        "complete_training_examples": int(data_protocol["examples_used_for_client_training"]),
        "internal_validation_examples": int(data_protocol["examples_used_for_validation"]),
        "official_test_examples": int(data_protocol["official_test_examples"]),
        "official_test_access_count": int(data_protocol["official_test_access_count"]),
        "scientific_status": SCIENTIFIC_STATUS,
        "selected_client_population_examples_over_rounds": sum(
            int(value["total_selected_examples"]) for value in rounds
        ),
        "local_training_examples_presented_over_rounds": sum(
            int(value["total_training_examples_presented"]) for value in rounds
        ),
    }
    if record["distribution"] != final.get("distribution") or record["alpha"] != final.get("partition_alpha"):
        raise ValueError("reported distribution is incompatible with the execution")
    for key in (
        "local_epochs",
        "total_clients",
        "participating_clients",
        "timesteps",
        "momentum",
        "weight_decay",
        "aggregation_weighting",
    ):
        if final.get(key) != record[key]:
            raise ValueError(f"execution field {key} is incompatible")
    return record


def summarize_published_fedsnn(manifest: str | Path, runs_root: str | Path, output_dir: str | Path) -> dict:
    tasks = load_published_fedsnn_manifest(manifest)
    root = Path(runs_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, int], list[Path]] = defaultdict(list)
    experiment_names = {task.experiment for task in tasks}
    if root.is_dir():
        for path in sorted(root.iterdir()):
            config_path = path / "resolved_config.yaml"
            if not path.is_dir() or not config_path.is_file():
                continue
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if config.get("name") in experiment_names and config.get("seed") in FEDERATED_SEEDS:
                grouped[(config["name"], int(config["seed"]))].append(path)

    findings: list[str] = []
    runs: list[dict] = []
    for task in tasks:
        candidates = grouped.get((task.experiment, task.seed), [])
        if len(candidates) != 1:
            findings.append(f"{task.experiment} seed {task.seed}: expected one execution, found {len(candidates)}")
            continue
        try:
            runs.append(_validated_run(task, candidates[0]))
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            findings.append(f"{task.experiment} seed {task.seed}: {error}")

    treatment_summaries = []
    for template_task in tasks[::3]:
        treatment_runs = sorted(
            [run for run in runs if run["experiment"] == template_task.experiment],
            key=lambda value: value["seed"],
        )
        metrics = {}
        if [run["seed"] for run in treatment_runs] == list(FEDERATED_SEEDS):
            for key in (
                "official_test_accuracy",
                "official_test_macro_f1",
                "signed_difference_percentage_points",
                "absolute_difference_percentage_points",
            ):
                metrics[key] = _stats([float(run[key]) for run in treatment_runs])
        treatment = PAPER_TREATMENTS[template_task.experiment]
        treatment_summaries.append(
            {
                "experiment": template_task.experiment,
                "distribution": treatment["distribution"],
                "alpha": treatment["alpha"],
                "seed_count": len(treatment_runs),
                "seeds_completed": [run["seed"] for run in treatment_runs],
                "completed": len(treatment_runs) == 3,
                "descriptive_paper_reference": treatment["descriptive_reference_accuracy"],
                "scientific_status": SCIENTIFIC_STATUS,
                "metrics": metrics,
                "runs": treatment_runs,
            }
        )

    completed = len(runs) == 6 and all(item["completed"] for item in treatment_summaries)
    summary = {
        "schema_version": 3,
        "valid": completed and not findings,
        "validation_findings": findings,
        "expected_task_count": 6,
        "completed_task_count": len(runs),
        "completed": completed and not findings,
        "treatments_pooled": False,
        "scientific_status": SCIENTIFIC_STATUS,
        "acceptance_reference_accuracy": None,
        "treatment_summaries": treatment_summaries,
    }
    (output / "published_fedsnn_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "published_fedsnn_summary.csv", runs)
    _write_markdown(output / "published_fedsnn_summary.md", summary)
    return summary


def _write_csv(path: Path, runs: list[dict]) -> None:
    fields = list(dict.fromkeys(key for run in runs for key in run))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(runs)


def _write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Fed-SNN Table I evaluation summary",
        "",
        f"Execution completion: **{str(summary['completed']).lower()}**",
        "",
        "Scientific status: **equivalence not established**. IID and non-IID treatments are not pooled.",
    ]
    for treatment in summary["treatment_summaries"]:
        alpha = "n/a" if treatment["alpha"] is None else str(treatment["alpha"])
        lines.extend(
            [
                "",
                f"## {treatment['experiment']}",
                "",
                f"Distribution: `{treatment['distribution']}`; alpha: `{alpha}`; "
                f"seeds completed: {treatment['seed_count']}/3.",
                "",
                "| Seed | Final round | Official test | Macro-F1 | Paper reference | Signed pp | Absolute pp |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for run in treatment["runs"]:
            lines.append(
                f"| {run['seed']} | {run['final_round']} | {run['official_test_accuracy']:.6g} | "
                f"{run['official_test_macro_f1']:.6g} | {run['descriptive_paper_reference']:.6g} | "
                f"{run['signed_difference_percentage_points']:.6g} | "
                f"{run['absolute_difference_percentage_points']:.6g} |"
            )
        if treatment["runs"]:
            settings = treatment["runs"][0]
            lines.extend(
                [
                    "",
                    f"Settings: local epochs `{settings['local_epochs']}`, clients "
                    f"`{settings['total_clients']}/{settings['participating_clients']}`, timesteps "
                    f"`{settings['timesteps']}`, momentum `{settings['momentum']}`, weight decay "
                    f"`{settings['weight_decay']}`, aggregation `{settings['aggregation_weighting']}`.",
                    "",
                    "| Training examples | Internal validation | Official test | Official-test accesses | Status |",
                    "|---:|---:|---:|---:|---|",
                    f"| {settings['complete_training_examples']} | {settings['internal_validation_examples']} | "
                    f"{settings['official_test_examples']} | {settings['official_test_access_count']} | "
                    f"{settings['scientific_status']} |",
                ]
            )
    if summary["validation_findings"]:
        lines.extend(["", "## Validation findings", ""])
        lines.extend(f"- {value}" for value in summary["validation_findings"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Fed-SNN Table I evaluations.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = summarize_published_fedsnn(args.manifest, args.runs_root, args.output_dir)
    if not summary["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
