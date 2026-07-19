"""Write required resource figures and their exact source tables."""

from __future__ import annotations

import ast
import csv
import json
import os
from io import StringIO
from pathlib import Path

from fedapfa.utilities.serialization import atomic_write_text


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, stream.getvalue())


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "path.simplify": False,
        }
    )
    return plt


def _save_figure(figure, path: Path) -> None:
    pending = path.with_suffix(path.suffix + ".part")
    figure.savefig(pending, format="png")
    os.replace(pending, path)


def _scatter(rows, x, y, x_label, y_label, path, diagonal=False):
    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(5.4, 4.0), constrained_layout=True)
    for dataset, marker in (("shd", "o"), ("ssc", "x")):
        selected = [row for row in rows if row.get("dataset") == dataset]
        axis.scatter(
            [float(row[x]) for row in selected],
            [float(row[y]) for row in selected],
            s=9,
            alpha=0.55,
            marker=marker,
            label=dataset.upper(),
        )
    if diagonal and rows:
        limits = [min(float(row[x]) for row in rows), max(float(row[x]) for row in rows)]
        axis.plot(limits, limits, color="black", linewidth=1)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.legend()
    _save_figure(figure, path)
    plt.close(figure)


def _selected_prediction_rows(artifacts: Path, target: str, model_name: str) -> list[dict]:
    rows = _read_csv(artifacts / f"figure_source_{target}.csv")
    return [row for row in rows if row["setting"] == "joint" and row["model"] == model_name]


def _prediction_figures(artifacts: Path, root: Path, scheduler_name: str, energy_name: str) -> None:
    runtime = _selected_prediction_rows(artifacts, "client_wall_time_seconds", scheduler_name)
    energy = _selected_prediction_rows(artifacts, "gross_energy_joules", energy_name)
    specifications = [
        (
            "predicted_versus_measured_runtime",
            runtime,
            "measured",
            "predicted",
            "Measured client wall time (s)",
            "Predicted client wall time (s)",
            True,
        ),
        (
            "predicted_versus_measured_gross_energy",
            energy,
            "measured",
            "predicted",
            "Measured gross device energy (J)",
            "Predicted gross device energy (J)",
            True,
        ),
        (
            "runtime_residual_versus_example_count",
            runtime,
            "example_count",
            "residual",
            "Client examples",
            "Runtime residual (s)",
            False,
        ),
        (
            "runtime_residual_versus_input_event_count",
            runtime,
            "input_event_count",
            "residual",
            "Raw input events",
            "Runtime residual (s)",
            False,
        ),
        (
            "runtime_residual_versus_padded_timesteps",
            runtime,
            "padded_time_bins",
            "residual",
            "Estimated padded time bins",
            "Runtime residual (s)",
            False,
        ),
    ]
    for name, rows, x, y, x_label, y_label, diagonal in specifications:
        _write_csv(root / f"source_{name}.csv", rows)
        _scatter(rows, x, y, x_label, y_label, root / f"{name}.png", diagonal)


def _comparison_figures(artifacts: Path, root: Path) -> None:
    evaluation = json.loads((artifacts / "cost_model_evaluation.json").read_text(encoding="utf-8"))
    source = _read_csv(artifacts / "figure_source_client_wall_time_seconds.csv")
    joint = [row for row in source if row["setting"] == "joint"]
    errors = [
        {
            "dataset": row["dataset"],
            "model": row["model"],
            "absolute_error_seconds": abs(float(row["residual"])),
        }
        for row in joint
    ]
    _write_csv(root / "source_error_distribution_by_model.csv", errors)
    plt = _matplotlib()
    models = ["constant", "size", "event_structure", "historical_spike", "diagnostic_oracle"]
    figure, axis = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    axis.boxplot(
        [[value["absolute_error_seconds"] for value in errors if value["model"] == model] for model in models],
        tick_labels=models,
        showfliers=False,
    )
    axis.tick_params(axis="x", rotation=25)
    axis.set_ylabel("Absolute runtime error (s)")
    _save_figure(figure, root / "error_distribution_by_model.png")
    plt.close(figure)
    ablation = []
    target = evaluation["settings"]["joint"]["targets"]["client_wall_time_seconds"]
    for model in models:
        for dataset, metrics in target[model]["evaluation"]["per_dataset"].items():
            ablation.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "median_absolute_error_seconds": metrics["median_absolute_error"],
                }
            )
    _write_csv(root / "source_feature_ablation_comparison.csv", ablation)
    figure, axis = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    width = 0.35
    x_values = list(range(len(models)))
    for offset, dataset in ((-width / 2, "shd"), (width / 2, "ssc")):
        axis.bar(
            [value + offset for value in x_values],
            [
                next(
                    float(row["median_absolute_error_seconds"])
                    for row in ablation
                    if row["model"] == model and row["dataset"] == dataset
                )
                for model in models
            ],
            width,
            label=dataset.upper(),
        )
    axis.set_xticks(x_values, models, rotation=25)
    axis.set_ylabel("Median absolute runtime error (s)")
    axis.legend()
    _save_figure(figure, root / "feature_ablation_comparison.png")
    plt.close(figure)
    transfer = []
    for setting in ("shd_to_ssc_transfer", "ssc_to_shd_transfer"):
        record = evaluation["settings"][setting]["targets"]["client_wall_time_seconds"]
        for model in models:
            transfer.append(
                {
                    "setting": setting,
                    "model": model,
                    "mean_absolute_error_seconds": record[model]["evaluation"]["joint"]["mean_absolute_error"],
                }
            )
    _write_csv(root / "source_cross_dataset_transfer_comparison.csv", transfer)
    figure, axis = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    for setting in ("shd_to_ssc_transfer", "ssc_to_shd_transfer"):
        axis.plot(
            models,
            [
                float(row["mean_absolute_error_seconds"])
                for model in models
                for row in transfer
                if row["setting"] == setting and row["model"] == model
            ],
            marker="o",
            label=setting,
        )
    axis.tick_params(axis="x", rotation=25)
    axis.set_ylabel("Mean absolute runtime error (s)")
    axis.legend()
    _save_figure(figure, root / "cross_dataset_transfer_comparison.png")
    plt.close(figure)


def _power_figure(run_dir: Path, root: Path) -> None:
    with (run_dir / "device_samples.jsonl").open("r", encoding="utf-8") as handle:
        samples = [json.loads(line) for line in handle if line.strip()]
    with (run_dir / "execution_intervals.jsonl").open("r", encoding="utf-8") as handle:
        intervals = [json.loads(line) for line in handle if line.strip()]
    clients = [value for value in intervals if value["category"] == "client_training" and value["accepted"]][:5]
    if not clients:
        raise ValueError("representative power trace requires accepted client intervals")
    lower = clients[0]["start_ns"]
    upper = clients[-1]["end_ns"]
    source = [
        {
            "record_kind": "sample",
            "seconds_from_start": (value["monotonic_timestamp_ns"] - lower) / 1_000_000_000,
            "power_watts": value["power_watts"],
            "interval_id": "",
        }
        for value in samples
        if lower <= value["monotonic_timestamp_ns"] <= upper and value["power_watts"] is not None
    ]
    source.extend(
        {
            "record_kind": boundary,
            "seconds_from_start": (value[f"{boundary}_ns"] - lower) / 1_000_000_000,
            "power_watts": "",
            "interval_id": value["interval_id"],
        }
        for value in clients
        for boundary in ("start", "end")
    )
    _write_csv(root / "source_representative_power_trace.csv", source)
    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    values = [row for row in source if row["record_kind"] == "sample"]
    axis.plot(
        [row["seconds_from_start"] for row in values],
        [row["power_watts"] for row in values],
        linewidth=1,
    )
    for row in source:
        if row["record_kind"] in {"start", "end"}:
            axis.axvline(float(row["seconds_from_start"]), color="black", alpha=0.25, linewidth=0.7)
    axis.set_xlabel("Seconds from selected trace start")
    axis.set_ylabel("Device power (W)")
    _save_figure(figure, root / "representative_power_trace.png")
    plt.close(figure)


def _assignment_figure(artifacts: Path, root: Path, scheduler_name: str) -> None:
    source = _read_csv(artifacts / "figure_source_assignment_load.csv")
    assignment_rows = [
        value
        for row in source
        if row["cost_model"] == scheduler_name
        and row["strategy"] == "predicted_cost_longest_first"
        and row["process_count"] == "4"
        for value in (
            {
                "dataset": row["dataset"],
                "scientific_seed": row["scientific_seed"],
                "communication_round": row["communication_round"],
                "process_index": process_index,
                "predicted_process_load_seconds": predicted,
                "measured_process_load_seconds": measured,
            }
            for process_index, (predicted, measured) in enumerate(
                zip(
                    ast.literal_eval(row["predicted_process_loads_seconds"]),
                    ast.literal_eval(row["measured_process_loads_seconds"]),
                    strict=True,
                )
            )
        )
    ]
    _write_csv(root / "source_observed_load_versus_predicted_assignment_load.csv", assignment_rows)
    _scatter(
        assignment_rows,
        "predicted_process_load_seconds",
        "measured_process_load_seconds",
        "Predicted process load (s)",
        "Observed process load (s)",
        root / "observed_load_versus_predicted_assignment_load.png",
        True,
    )


def write_resource_figures(
    artifact_root: str | Path, result_root: str | Path, run_directories: list[str | Path]
) -> list[str]:
    artifacts = Path(artifact_root).resolve()
    root = Path(result_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    evaluation = json.loads((artifacts / "cost_model_evaluation.json").read_text(encoding="utf-8"))
    scheduler_name = evaluation["scheduler_model"]
    energy_name = evaluation["energy_model"]
    _prediction_figures(artifacts, root, scheduler_name, energy_name)
    _comparison_figures(artifacts, root)
    _power_figure(sorted(Path(value).resolve() for value in run_directories)[0], root)
    _assignment_figure(artifacts, root, scheduler_name)
    return sorted(value.name for value in root.glob("*.png"))
