"""Deterministic figures from committed comparative summary artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCALING_ORDER = (
    "one_node_one_gpu",
    "one_node_two_gpu",
    "one_node_four_gpu",
    "two_nodes_four_gpus",
)
NON_IID_ORDER = (
    "iid",
    "dirichlet_alpha_1_0",
    "dirichlet_alpha_0_5",
    "dirichlet_alpha_0_1",
)
NON_IID_LABELS = ("IID", "α=1.0", "α=0.5", "α=0.1")


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, metadata={"Software": "fedapfa"})
    plt.close()


def _paired_lines(path: Path, records: list[dict], field: str, ylabel: str, title: str, order) -> None:
    plt.figure(figsize=(7.4, 4.3))
    for dataset, marker in (("shd", "o"), ("ssc", "s")):
        for treatment in order:
            selected = sorted(
                (value for value in records if value["dataset"] == dataset and value["treatment_id"] == treatment),
                key=lambda value: value["seed"],
            )
            if selected:
                plt.plot(
                    [value["seed"] for value in selected],
                    [value[field] for value in selected],
                    marker=marker,
                    label=f"{dataset.upper()} {treatment}",
                )
    plt.xlabel("Scientific seed")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=6)
    _save(path)


def _group_lines(path: Path, groups: list[dict], field: str, ylabel: str, title: str, order, labels) -> None:
    plt.figure(figsize=(7.4, 4.3))
    for dataset, marker in (("shd", "o"), ("ssc", "s")):
        selected = {value["treatment_id"]: value for value in groups if value["dataset"] == dataset}
        means = [selected[value]["statistics"][field]["mean"] for value in order]
        deviations = [selected[value]["statistics"][field]["sample_standard_deviation"] for value in order]
        plt.errorbar(labels, means, yerr=deviations, marker=marker, capsize=3, label=dataset.upper())
    plt.xlabel("Treatment")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    _save(path)


def generate_system_scaling_figures(summary_path: str | Path, output_dir: str | Path) -> list[Path]:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if summary.get("valid") is not True or summary.get("collection") != "system_scaling_energy_evaluation":
        raise ValueError("scaling figures require a valid system-scaling summary")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    groups = summary["groups"]
    runs = summary["runs"]
    paired = summary["paired_records"]
    generated = []
    specifications = (
        ("runtime_by_dataset_and_topology", "runtime_seconds", "Runtime (s)", "Runtime by topology"),
        ("parallel_efficiency", "parallel_efficiency", "Parallel efficiency", "Parallel efficiency"),
        ("gpu_utilization", "gpu_utilization_percent", "GPU utilization (%)", "GPU utilization"),
        ("load_imbalance", "load_imbalance", "Load imbalance (fraction)", "Load imbalance"),
    )
    labels = tuple(value.replace("_", "\n") for value in SCALING_ORDER)
    for filename, field, ylabel, title in specifications:
        path = output / f"scaling_{filename}.png"
        _group_lines(path, groups, field, ylabel, title, SCALING_ORDER, labels)
        generated.append(path)
    path = output / "scaling_paired_speedup_by_seed.png"
    _paired_lines(path, paired, "speedup", "Speedup (T₁/Tₚ)", "Paired speedup by seed", SCALING_ORDER)
    generated.append(path)
    plt.figure(figsize=(7.4, 4.3))
    for dataset, marker in (("shd", "o"), ("ssc", "s")):
        selected = {value["treatment_id"]: value for value in groups if value["dataset"] == dataset}
        gross = [selected[value]["statistics"]["gross_execution_energy_joules"]["mean"] for value in SCALING_ORDER]
        dynamic = [
            selected[value]["statistics"]["idle_adjusted_execution_energy_joules"]["mean"] for value in SCALING_ORDER
        ]
        plt.plot(labels, gross, marker=marker, label=f"{dataset.upper()} gross")
        plt.plot(labels, dynamic, marker=marker, linestyle="--", label=f"{dataset.upper()} idle-adjusted")
    plt.xlabel("Topology")
    plt.ylabel("Energy (J)")
    plt.title("Gross and idle-adjusted energy by topology")
    plt.legend(fontsize=7)
    path = output / "scaling_energy_by_topology.png"
    _save(path)
    generated.append(path)
    plt.figure(figsize=(6.5, 4.5))
    for dataset, marker in (("shd", "o"), ("ssc", "s")):
        selected = [value for value in runs if value["dataset"] == dataset]
        plt.scatter(
            [value["runtime_seconds"] for value in selected],
            [value["gross_execution_energy_joules"] for value in selected],
            marker=marker,
            label=dataset.upper(),
        )
    plt.xlabel("Runtime (s)")
    plt.ylabel("Gross energy (J)")
    plt.title("Energy versus runtime")
    plt.legend()
    path = output / "scaling_energy_versus_runtime.png"
    _save(path)
    generated.append(path)
    numerical = summary["numerical_identity"]
    path = output / "scaling_numerical_parameter_difference.png"
    _paired_lines(
        path,
        numerical,
        "maximum_absolute_parameter_difference",
        "Maximum absolute parameter difference",
        "Numerical parameter difference",
        SCALING_ORDER,
    )
    generated.append(path)
    return generated


def generate_non_iid_figures(summary_path: str | Path, output_dir: str | Path) -> list[Path]:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if summary.get("valid") is not True or summary.get("collection") != "non_iid_energy_evaluation":
        raise ValueError("non-IID figures require a valid non-IID summary")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    groups = summary["groups"]
    runs = summary["runs"]
    generated = []
    specifications = (
        (
            "official_test_accuracy_by_alpha",
            "official_test_accuracy",
            "Official-test accuracy",
            "Accuracy by distribution",
        ),
        ("macro_f1_by_alpha", "macro_f1", "Macro-F1", "Macro-F1 by distribution"),
        ("runtime_by_alpha", "runtime_seconds", "Runtime (s)", "Runtime by distribution"),
        (
            "client_population_imbalance",
            "client_population_imbalance",
            "Population imbalance (fraction)",
            "Client-population imbalance",
        ),
        ("event_count_imbalance", "client_event_imbalance", "Event imbalance (fraction)", "Event-count imbalance"),
        ("load_imbalance", "load_imbalance", "Load imbalance (fraction)", "Execution load imbalance"),
    )
    for filename, field, ylabel, title in specifications:
        path = output / f"non_iid_{filename}.png"
        _group_lines(path, groups, field, ylabel, title, NON_IID_ORDER, NON_IID_LABELS)
        generated.append(path)
    plt.figure(figsize=(7.4, 4.3))
    for dataset, marker in (("shd", "o"), ("ssc", "s")):
        selected = {value["treatment_id"]: value for value in groups if value["dataset"] == dataset}
        gross = [selected[value]["statistics"]["gross_execution_energy_joules"]["mean"] for value in NON_IID_ORDER]
        dynamic = [
            selected[value]["statistics"]["idle_adjusted_execution_energy_joules"]["mean"] for value in NON_IID_ORDER
        ]
        plt.plot(NON_IID_LABELS, gross, marker=marker, label=f"{dataset.upper()} gross")
        plt.plot(
            NON_IID_LABELS,
            dynamic,
            marker=marker,
            linestyle="--",
            label=f"{dataset.upper()} idle-adjusted",
        )
    plt.xlabel("Distribution")
    plt.ylabel("Energy (J)")
    plt.title("Gross and idle-adjusted energy by distribution")
    plt.legend(fontsize=7)
    path = output / "non_iid_energy_by_alpha.png"
    _save(path)
    generated.append(path)
    plt.figure(figsize=(6.5, 4.5))
    for dataset, marker in (("shd", "o"), ("ssc", "s")):
        selected = [value for value in runs if value["dataset"] == dataset]
        plt.scatter(
            [value["gross_execution_energy_joules"] for value in selected],
            [value["official_test_accuracy"] for value in selected],
            marker=marker,
            label=dataset.upper(),
        )
    plt.xlabel("Gross energy (J)")
    plt.ylabel("Official-test accuracy")
    plt.title("Accuracy versus energy")
    plt.legend()
    path = output / "non_iid_accuracy_versus_energy.png"
    _save(path)
    generated.append(path)
    return generated
