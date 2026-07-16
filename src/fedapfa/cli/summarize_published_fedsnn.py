"""Validate and summarize independent published Fed-SNN protocol executions."""

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
        raise ValueError("published protocol statistics require three seeds")
    return {"mean": statistics.mean(values), "sample_standard_deviation": statistics.stdev(values)}


def summarize_published_fedsnn(manifest: str | Path, runs_root: str | Path, output_dir: str | Path) -> dict:
    tasks = load_published_fedsnn_manifest(manifest)
    root = Path(runs_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, int], list[Path]] = defaultdict(list)
    if root.is_dir():
        for path in sorted(root.iterdir()):
            config_path = path / "resolved_config.yaml"
            if not path.is_dir() or not config_path.is_file():
                continue
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if config.get("name") == tasks[0].experiment and config.get("seed") in FEDERATED_SEEDS:
                grouped[(config["name"], int(config["seed"]))].append(path)
    findings: list[str] = []
    runs = []
    for task in tasks:
        candidates = grouped.get((task.experiment, task.seed), [])
        if len(candidates) != 1:
            findings.append(f"seed {task.seed}: expected one execution, found {len(candidates)}")
            continue
        path = candidates[0]
        try:
            acceptance = _read_json(path / "acceptance.json")
            final = _read_json(path / "final_metrics.json")
            rounds = [
                json.loads(line)
                for line in (path / "round_metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if acceptance.get("completed") is not True or acceptance.get("accepted") is not True:
                raise ValueError("execution is not accepted and completed")
            if len(rounds) != 100:
                raise ValueError("communication-round records are incomplete")
            if acceptance.get("scientific_status") != "not_claimed":
                raise ValueError("scientific status must remain not_claimed without a target")
            if acceptance.get("reference_test_accuracy") is not None:
                raise ValueError("an unverified reference accuracy was configured")
            runs.append(
                {
                    "seed": task.seed,
                    "run_directory": str(path),
                    "completed": True,
                    "scientific_status": "not_claimed",
                    "best_validation_accuracy": _finite(final["best_validation_accuracy"], "best accuracy"),
                    "official_test_accuracy": _finite(final["test"]["accuracy"], "test accuracy"),
                    "macro_f1": _finite(final["test"]["macro_f1"], "macro-F1"),
                    "selected_round": _finite(final["selected_round"], "selected round"),
                    "spike_rates": final["test"]["spike_rates"],
                    "logical_communication_bytes": _finite(
                        final["logical_communication"]["cumulative_total_bytes"], "communication"
                    ),
                    "execution_time_seconds": _finite(final["execution_time_seconds"], "duration"),
                    "peak_cuda_memory_bytes": int(final["peak_cuda_memory_bytes"]),
                    "parameter_count": int(final["parameter_count"]),
                    "protocol_assumptions": final["protocol_assumptions"],
                    "reference_test_accuracy": acceptance.get("reference_test_accuracy"),
                    "tolerance": acceptance.get("tolerance"),
                }
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            findings.append(f"seed {task.seed}: {error}")
    runs.sort(key=lambda value: value["seed"])
    metrics = {}
    if [value["seed"] for value in runs] == list(FEDERATED_SEEDS):
        assumptions = {tuple(value["protocol_assumptions"]) for value in runs}
        parameter_counts = {value["parameter_count"] for value in runs}
        if len(assumptions) != 1:
            findings.append("protocol assumptions differ across seeds")
        if len(parameter_counts) != 1:
            findings.append("model parameter counts differ across seeds")
        for key in (
            "best_validation_accuracy",
            "official_test_accuracy",
            "macro_f1",
            "selected_round",
            "logical_communication_bytes",
            "execution_time_seconds",
            "peak_cuda_memory_bytes",
        ):
            metrics[key] = _stats([float(value[key]) for value in runs])
    summary = {
        "schema_version": 1,
        "valid": not findings and len(runs) == 3,
        "validation_findings": findings,
        "expected_task_count": 3,
        "seeds_completed": [value["seed"] for value in runs],
        "completed": len(runs) == 3 and not findings,
        "scientific_status": "not_claimed",
        "reference_test_accuracy": None,
        "tolerance": None,
        "protocol_assumptions": runs[0]["protocol_assumptions"] if runs else tasks[0].config["protocol_assumptions"],
        "model_parameter_count": runs[0]["parameter_count"] if len(runs) == 3 else None,
        "metrics": metrics,
        "runs": runs,
    }
    (output / "published_fedsnn_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_csv(output / "published_fedsnn_summary.csv", summary)
    _write_markdown(output / "published_fedsnn_summary.md", summary)
    return summary


def _write_csv(path: Path, summary: dict) -> None:
    rows = []
    for run in summary["runs"]:
        row = {key: value for key, value in run.items() if key not in {"spike_rates", "protocol_assumptions"}}
        row.update({f"spike_rate_{key}": value for key, value in run["spike_rates"].items()})
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Independent published Fed-SNN protocol summary",
        "",
        f"Execution completion: **{str(summary['completed']).lower()}**",
        "",
        "Scientific status: **not_claimed**. No verified numerical target is configured, "
        "so completion is not a reproduction pass.",
        "",
        "| Seed | Best validation accuracy | Official-test accuracy | Macro-F1 | Selected round |",
        "|---:|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        lines.append(
            f"| {run['seed']} | {run['best_validation_accuracy']:.6g} | {run['official_test_accuracy']:.6g} | "
            f"{run['macro_f1']:.6g} | {run['selected_round']:.0f} |"
        )
    lines.extend(["", "## Protocol assumptions", ""])
    lines.extend(f"- {value}" for value in summary["protocol_assumptions"])
    if summary["validation_findings"]:
        lines.extend(["", "## Validation findings", ""])
        lines.extend(f"- {value}" for value in summary["validation_findings"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize independent published Fed-SNN protocol executions.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = summarize_published_fedsnn(args.manifest, args.runs_root, args.output_dir)
    if not summary["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
