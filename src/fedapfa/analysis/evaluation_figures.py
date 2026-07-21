"""Deterministic figures produced only after validated summary evidence exists."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, metadata={"Software": "fedapfa"})
    plt.close()


def _scatter(path: Path, x, y, xlabel: str, ylabel: str, title: str) -> None:
    plt.figure(figsize=(6.4, 4.2))
    plt.scatter(x, y, s=14, alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    _save(path)


def _paired_lines(path: Path, records: list[dict], metric: str, ylabel: str, title: str) -> None:
    plt.figure(figsize=(7.2, 4.2))
    for dataset in ("shd", "ssc"):
        selected = [value for value in records if value["dataset"] == dataset]
        plt.plot(
            [value["seed"] for value in selected],
            [value[metric] for value in selected],
            marker="o",
            label=dataset.upper(),
        )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Scientific seed")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    _save(path)


def generate_evaluation_figures(summary_path: str | Path, output_dir: str | Path) -> list[Path]:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if summary.get("valid") is not True:
        raise ValueError("figures require a valid completed summary")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated = []
    collection = summary["collection"]
    runs = summary["runs"]
    pairs = summary["paired_records"]
    if collection == "scheduling_evaluation":
        for dataset in ("shd", "ssc"):
            dataset_runs = [value for value in runs if value["dataset"] == dataset]
            plt.figure(figsize=(6.4, 4.2))
            event_runs = [
                value for value in dataset_runs if value["strategy"] == "event_structure_longest_processing_time"
            ]
            for run in sorted(event_runs, key=lambda value: value["seed"]):
                predicted = []
                observed = []
                for round_record in run["round_measurements"]:
                    for assignment in round_record["client_assignments"]:
                        predicted.append(assignment["cost"])
                        observed.append(assignment["actual_client_wall_duration_seconds"])
                plt.scatter(
                    predicted,
                    observed,
                    s=14,
                    alpha=0.75,
                    label=f"seed {run['seed']}",
                )
            path = output / f"{dataset}_predicted_versus_observed_client_time.png"
            plt.xlabel("Predicted client time (s)")
            plt.ylabel("Observed client time (s)")
            plt.title(f"{dataset.upper()} predicted and observed client time")
            plt.legend()
            _save(path)
            generated.append(path)

            for field, ylabel, name in (
                ("total_round_time_seconds", "Round makespan (s)", "round_makespan"),
                ("process_load_imbalance", "Observed load imbalance (fraction)", "load_imbalance"),
                ("scheduler_overhead_fraction", "Scheduler overhead (fraction)", "scheduler_overhead_fraction"),
            ):
                plt.figure(figsize=(7.2, 4.2))
                for run in sorted(dataset_runs, key=lambda value: (value["strategy"], value["seed"])):
                    values = run["round_measurements"]
                    plt.plot(
                        [value["round_number"] for value in values],
                        [value[field] for value in values],
                        label=f"{run['strategy']}, seed {run['seed']}",
                    )
                plt.xlabel("Communication round")
                plt.ylabel(ylabel)
                plt.title(f"{dataset.upper()} {name.replace('_', ' ')}")
                plt.legend(fontsize=7)
                path = output / f"{dataset}_{name}.png"
                _save(path)
                generated.append(path)

            event_run = next(
                value
                for value in dataset_runs
                if value["strategy"] == "event_structure_longest_processing_time" and value["seed"] == 37
            )
            ranks = sorted(event_run["round_measurements"][0]["process_busy_time_seconds"], key=int)
            plt.figure(figsize=(8.0, 4.2))
            for rank in ranks:
                plt.plot(
                    [value["round_number"] for value in event_run["round_measurements"]],
                    [value["process_busy_time_seconds"][rank] for value in event_run["round_measurements"]],
                    label=f"rank {rank}",
                )
            plt.xlabel("Communication round")
            plt.ylabel("Assigned process busy time (s)")
            plt.title(f"{dataset.upper()} process workload timeline, seed 37")
            plt.legend()
            path = output / f"{dataset}_process_workload_timeline.png"
            _save(path)
            generated.append(path)

            plt.figure(figsize=(7.2, 4.2))
            for strategy in sorted({value["strategy"] for value in dataset_runs}):
                selected = [value for value in dataset_runs if value["strategy"] == strategy]
                plt.plot(
                    [value["seed"] for value in selected],
                    [value["gpu_utilization_percent"] for value in selected],
                    marker="o",
                    label=strategy,
                )
            plt.xlabel("Scientific seed")
            plt.ylabel("GPU utilization (%)")
            plt.title(f"{dataset.upper()} GPU utilization")
            plt.legend(fontsize=7)
            path = output / f"{dataset}_gpu_utilization.png"
            _save(path)
            generated.append(path)
        event_pairs = [value for value in pairs if value["treatment"] == "event_structure_longest_processing_time"]
        path = output / "paired_runtime_reduction_by_dataset_and_seed.png"
        _paired_lines(
            path,
            event_pairs,
            "paired_runtime_reduction",
            "Runtime reduction (fraction)",
            "Event-structure scheduler paired runtime reduction",
        )
        generated.append(path)
    else:
        for dataset in ("shd", "ssc"):
            dataset_runs = [value for value in runs if value["dataset"] == dataset]
            for metric, ylabel, name in (
                ("aggregation_time_seconds", "Aggregation time (s)", "aggregation_time"),
                ("logical_inter_node_bytes", "Logical inter-node movement (bytes)", "logical_inter_node_movement"),
            ):
                plt.figure(figsize=(7.2, 4.2))
                for topology in ("flat_ordered", "node_hierarchical"):
                    selected = [value for value in dataset_runs if value["aggregation_topology"] == topology]
                    plt.plot(
                        [value["seed"] for value in selected],
                        [value[metric] for value in selected],
                        marker="o",
                        label=topology,
                    )
                plt.xlabel("Scientific seed")
                plt.ylabel(ylabel)
                plt.title(f"{dataset.upper()} {name.replace('_', ' ')}")
                plt.legend()
                path = output / f"{dataset}_{name}.png"
                _save(path)
                generated.append(path)
            hierarchical = [value for value in dataset_runs if value["aggregation_topology"] == "node_hierarchical"]
            waiting = [
                sum(
                    sum(record["estimated_idle_time_seconds_by_process_rank"].values())
                    for record in value["round_measurements"]
                )
                for value in hierarchical
            ]
            path = output / f"{dataset}_rank_waiting_time.png"
            _scatter(
                path,
                [value["seed"] for value in hierarchical],
                waiting,
                "Scientific seed",
                "Summed rank waiting time (s)",
                f"{dataset.upper()} hierarchical rank waiting",
            )
            generated.append(path)
        path = output / "hierarchical_maximum_parameter_difference.png"
        _paired_lines(
            path,
            pairs,
            "maximum_absolute_parameter_difference",
            "Maximum absolute parameter difference",
            "Hierarchical parameter difference",
        )
        generated.append(path)
        path = output / "hierarchical_paired_runtime_difference.png"
        _paired_lines(
            path,
            pairs,
            "runtime_regression_fraction",
            "Runtime regression (fraction)",
            "Hierarchical paired runtime difference",
        )
        generated.append(path)
    return generated
