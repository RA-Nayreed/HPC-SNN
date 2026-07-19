"""Cost-model fitting, evaluation, export, and summary artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from fedapfa.utilities.serialization import atomic_write_json, atomic_write_text

from .assignment import evaluate_round_assignments
from .dataset import validate_accepted_run
from .decision import decide_spike_history, ensure_exportable
from .evaluation import evaluate_predictions, regression_metrics
from .history import add_causal_history
from .regression import CostModel, fit_regression
from .splits import evaluation_settings, grouped_fit_validation, row_identity

TARGETS = (
    "client_wall_time_seconds",
    "cuda_event_time_seconds",
    "gross_energy_joules",
    "idle_adjusted_energy_joules",
)
SIZE_FEATURES = ["example_count", "local_batch_count"]
EVENT_FEATURES = [
    *SIZE_FEATURES,
    "total_raw_input_events",
    "mean_sequence_length",
    "median_sequence_length",
    "maximum_sequence_length",
    "total_valid_time_bins",
    "estimated_padded_time_bins",
    "padding_fraction",
    "event_density",
]
HISTORY_FEATURES = [
    *EVENT_FEATURES,
    "previous_wall_duration",
    "previous_gross_energy",
    "previous_idle_adjusted_energy",
    "previous_layer1_spike_rate",
    "previous_layer2_spike_rate",
    "previous_spikes_per_example",
    "exponentially_weighted_duration",
    "exponentially_weighted_gross_energy",
    "exponentially_weighted_idle_adjusted_energy",
    "exponentially_weighted_layer1_spike_rate",
    "exponentially_weighted_layer2_spike_rate",
    "historical_observation_count",
    "missing_previous_wall_duration",
    "missing_previous_gross_energy",
    "missing_previous_idle_adjusted_energy",
    "missing_previous_layer1_spike_rate",
    "missing_previous_layer2_spike_rate",
    "missing_previous_spikes_per_example",
]
ORACLE_FEATURES = [
    *EVENT_FEATURES,
    "layer1_spike_count",
    "layer2_spike_count",
    "layer1_spike_rate",
    "layer2_spike_rate",
]
MODEL_FEATURES = {
    "constant": [],
    "size": SIZE_FEATURES,
    "event_structure": EVENT_FEATURES,
    "historical_spike": HISTORY_FEATURES,
    "diagnostic_oracle": ORACLE_FEATURES,
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_slurm_accounting(path: Path, allocation_ids: str | set[str]) -> dict:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        required = {"JobID", "State", "ExitCode", "ElapsedRaw", "AllocTRES", "Start", "End"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Slurm accounting fields are incomplete")
        records = [dict(value) for value in reader if value.get("JobID")]
    requested = {allocation_ids} if isinstance(allocation_ids, str) else set(allocation_ids)
    if not requested:
        raise ValueError("Slurm accounting allocation identities are empty")
    matched_records = []
    for allocation_id in sorted(requested):
        matches = [value for value in records if value["JobID"] == allocation_id]
        if len(matches) != 1:
            raise ValueError("Slurm accounting does not contain exactly one record per allocation")
        record = matches[0]
        state = record["State"].split("+", 1)[0]
        completed = state == "COMPLETED" and record["ExitCode"] == "0:0"
        interrupted = state in {"CANCELLED", "PREEMPTED", "TIMEOUT"}
        if not completed and not interrupted:
            raise ValueError("Slurm allocation did not complete or record a resumable interruption")
        try:
            elapsed = int(record["ElapsedRaw"])
        except (TypeError, ValueError) as error:
            raise ValueError("Slurm allocation elapsed time is invalid") from error
        tres = {}
        for item in record["AllocTRES"].split(","):
            if "=" in item:
                name, value = item.split("=", 1)
                tres[name] = value
        gpu_count_value = tres.get("gres/gpu")
        if gpu_count_value is None:
            typed = [value for name, value in tres.items() if name.startswith("gres/gpu:")]
            gpu_count_value = typed[0] if len(typed) == 1 else None
        try:
            gpu_count = int(gpu_count_value)
        except (TypeError, ValueError) as error:
            raise ValueError("Slurm accounting GPU allocation is invalid") from error
        if gpu_count != 1 or elapsed <= 0 or not record["Start"] or not record["End"]:
            raise ValueError("Slurm accounting allocation values are incompatible")
        matched_records.append(
            {
                "job_id": allocation_id,
                "state": record["State"],
                "exit_code": record["ExitCode"],
                "elapsed_seconds": elapsed,
                "allocated_gpu_count": gpu_count,
                "allocated_gpu_hours": elapsed * gpu_count / 3600.0,
                "allocated_tres": record["AllocTRES"],
                "start": record["Start"],
                "end": record["End"],
                "resumable_interruption": interrupted,
            }
        )
    if not any(not value["resumable_interruption"] for value in matched_records):
        raise ValueError("Slurm accounting lacks a completed allocation")
    return {
        "allocation_ids": sorted(requested),
        "allocation_records": matched_records,
        "elapsed_seconds": sum(value["elapsed_seconds"] for value in matched_records),
        "allocated_gpu_hours": sum(value["allocated_gpu_hours"] for value in matched_records),
        "pending_time_seconds": None,
        "device_energy_inferred_from_allocation": False,
        "source_sha256": _file_sha256(path),
    }


def load_cost_csv(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            row = {}
            for key, value in source.items():
                if value == "":
                    row[key] = None
                elif value in {"True", "False"}:
                    row[key] = value == "True"
                else:
                    try:
                        number = float(value)
                        row[key] = int(number) if number.is_integer() else number
                    except ValueError:
                        row[key] = value
            rows.append(row)
    return rows


def _candidate_models(
    fitting: list[dict],
    validation: list[dict],
    target: str,
    model_name: str,
    penalties: list[float],
    provenance: dict,
) -> tuple[CostModel, dict]:
    features = MODEL_FEATURES[model_name]
    specifications = [("median", 0.0, "identity")] if model_name == "constant" else []
    if model_name != "constant":
        for transformation in ("identity", "log"):
            if transformation == "log" and any(float(row[target]) <= 0 for row in fitting + validation):
                continue
            specifications.append(("robust", 0.0, transformation))
            specifications.extend(("ridge", float(value), transformation) for value in penalties)
    decisions = []
    best = None
    for family, penalty, transformation in specifications:
        model = fit_regression(
            fitting,
            target,
            features,
            family,
            penalty,
            transformation,
            software_provenance=provenance,
        )
        metrics = regression_metrics(
            [float(row[target]) for row in validation],
            model.predict(validation),
            float(provenance["percentage_denominator_floor"]),
        )
        record = {
            "family": family,
            "regularization": penalty,
            "target_transformation": transformation,
            "validation_metrics": metrics,
        }
        decisions.append(record)
        key = (metrics["mean_absolute_error"], family, penalty, transformation)
        if best is None or key < best[0]:
            best = (key, record)
    selected = best[1]
    decision = {
        "selection_metric": "mean_absolute_error",
        "candidates": decisions,
        "selected": selected,
        "selection_rows": [row_identity(row) for row in validation],
    }
    model = fit_regression(
        fitting + validation,
        target,
        features,
        selected["family"],
        float(selected["regularization"]),
        selected["target_transformation"],
        decision,
        provenance,
    )
    return model, decision


def _select_history_coefficient(
    rows: list[dict],
    candidates: list[float],
    penalties: list[float],
    provenance: dict,
) -> tuple[float, dict]:
    records = []
    for coefficient in candidates:
        candidate_rows = add_causal_history(rows, coefficient)
        fit_collection = [
            value for value in candidate_rows if int(value["scientific_seed"]) in {7, 17}
        ]
        fitting, validation = grouped_fit_validation(fit_collection)
        if any(int(value["scientific_seed"]) == 27 for value in fitting + validation):
            raise RuntimeError("seed 27 entered historical-weight selection")
        _, decision = _candidate_models(
            fitting,
            validation,
            "client_wall_time_seconds",
            "historical_spike",
            penalties,
            provenance,
        )
        metric = float(decision["selected"]["validation_metrics"]["mean_absolute_error"])
        records.append(
            {
                "coefficient": coefficient,
                "validation_mean_absolute_error": metric,
                "fitting_row_identities": [row_identity(value) for value in fitting],
                "validation_row_identities": [row_identity(value) for value in validation],
            }
        )
    selected = min(records, key=lambda value: (value["validation_mean_absolute_error"], value["coefficient"]))
    return float(selected["coefficient"]), {
        "selection_metric": "client_wall_time_mean_absolute_error",
        "seed_27_used": False,
        "candidates": records,
        "selected_coefficient": float(selected["coefficient"]),
    }


def _write_table(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    from io import StringIO

    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, stream.getvalue())


def _metric_table_rows(setting: str, target: str, model_name: str, metrics: dict) -> list[dict]:
    records = []
    groups = [("joint", "all", metrics["joint"])]
    for name in (
        "per_dataset",
        "per_seed",
        "by_client_size_quartile",
        "by_sequence_length_quartile",
        "by_communication_round_interval",
    ):
        groups.extend((name, value, record) for value, record in metrics[name].items())
    groups.extend(
        (name, "all", metrics[name]) for name in ("without_history", "with_history")
    )
    for dimension, value, record in groups:
        records.append(
            {
                "setting": setting,
                "target": target,
                "model": model_name,
                "group_dimension": dimension,
                "group_value": value,
                **record,
            }
        )
    return records


def _roundtrip(model: CostModel, path: Path, rows: list[dict]) -> None:
    model.save(path)
    restored = CostModel.load(path)
    if not np.allclose(model.predict(rows), restored.predict(rows), rtol=0.0, atol=1e-12):
        raise RuntimeError("JSON model round trip changed predictions")


def fit_client_cost_models(rows: list[dict], result_root: str | Path, cost_config: dict, provenance: dict) -> dict:
    """Select with client-grouped validation and evaluate only after choices are frozen."""

    root = Path(result_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(int(row["scientific_seed"]) not in {7, 17, 27} for row in rows):
        raise ValueError("cost data contains an unexpected seed")
    software = {
        **provenance,
        "percentage_denominator_floor": float(cost_config["percentage_denominator_floor"]),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    history_coefficient, history_selection = _select_history_coefficient(
        rows,
        [float(value) for value in cost_config["historical_weight_candidates"]],
        [float(value) for value in cost_config["ridge_regularization"]],
        software,
    )
    software["historical_weight_selection"] = history_selection
    rows = add_causal_history(rows, history_coefficient)
    settings = evaluation_settings(rows)
    evaluations = {}
    fitted_models: dict[tuple[str, str, str], CostModel] = {}
    metric_rows = []
    prediction_sources: dict[str, list[dict]] = defaultdict(list)
    for setting, (fit_collection, evaluation_collection) in settings.items():
        fitting, validation = grouped_fit_validation(fit_collection)
        if any(int(row["scientific_seed"]) == 27 for row in fitting + validation):
            raise RuntimeError("seed 27 entered fitting or model selection")
        setting_record = {
            "fitting_row_identities": [row_identity(row) for row in fitting],
            "validation_row_identities": [row_identity(row) for row in validation],
            "evaluation_row_identities": [row_identity(row) for row in evaluation_collection],
            "targets": {},
        }
        for target in TARGETS:
            target_record = {}
            for model_name in MODEL_FEATURES:
                model, selection = _candidate_models(
                    fitting,
                    validation,
                    target,
                    model_name,
                    list(cost_config["ridge_regularization"]),
                    software,
                )
                fitted_models[(setting, target, model_name)] = model
                predictions = model.predict(evaluation_collection)
                metrics = evaluate_predictions(
                    evaluation_collection,
                    target,
                    predictions,
                    float(cost_config["percentage_denominator_floor"]),
                )
                model_path = root / "models" / setting / target / f"{model_name}.json"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                _roundtrip(model, model_path, evaluation_collection)
                target_record[model_name] = {
                    "model_path": str(model_path.relative_to(root)),
                    "feature_availability": (
                        "after_current_execution" if model_name == "diagnostic_oracle" else "before_assignment"
                    ),
                    "selection": selection,
                    "evaluation": metrics,
                    "json_roundtrip_verified": True,
                }
                metric_rows.extend(_metric_table_rows(setting, target, model_name, metrics))
                for row, prediction in zip(evaluation_collection, predictions, strict=True):
                    prediction_sources[target].append(
                        {
                            "setting": setting,
                            "model": model_name,
                            "dataset": row["dataset"],
                            "scientific_seed": row["scientific_seed"],
                            "communication_round": row["communication_round"],
                            "selected_position": row["selected_position"],
                            "measured": row[target],
                            "predicted": float(prediction),
                            "residual": float(prediction) - float(row[target]),
                            "example_count": row["example_count"],
                            "input_event_count": row["total_raw_input_events"],
                            "padded_time_bins": row["estimated_padded_time_bins"],
                        }
                    )
            setting_record["targets"][target] = target_record
        evaluations[setting] = setting_record
    within_metrics = {}
    for dataset, setting in (("shd", "shd_within_dataset"), ("ssc", "ssc_within_dataset")):
        target_record = evaluations[setting]["targets"]["client_wall_time_seconds"]
        non_spike_name = min(
            ("size", "event_structure"),
            key=lambda name: target_record[name]["selection"]["selected"]["validation_metrics"][
                "mean_absolute_error"
            ],
        )
        within_metrics[dataset] = {
            "historical_spike": target_record["historical_spike"]["evaluation"]["joint"],
            "strongest_non_spike": target_record[non_spike_name]["evaluation"]["joint"],
            "strongest_non_spike_name": non_spike_name,
        }
    joint_runtime = evaluations["joint"]["targets"]["client_wall_time_seconds"]
    joint_non_spike = min(
        ("size", "event_structure"),
        key=lambda name: joint_runtime[name]["selection"]["selected"]["validation_metrics"]["mean_absolute_error"],
    )
    evaluation_seed_rows = [row for row in rows if int(row["scientific_seed"]) == 27]
    assignment_records = []
    assignment_regret = {}
    prediction_times = {}
    for model_name in (joint_non_spike, "historical_spike"):
        model = fitted_models[("joint", "client_wall_time_seconds", model_name)]
        started = time.perf_counter_ns()
        model.predict(evaluation_seed_rows)
        prediction_times[model_name] = (time.perf_counter_ns() - started) / 1_000_000_000
        all_predictions = model.predict(rows)
        model_records = []
        for process_count in (2, 4):
            model_records.extend(evaluate_round_assignments(rows, all_predictions.tolist(), process_count))
        for value in model_records:
            value["cost_model"] = model_name
            value["data_role"] = (
                "untouched_evaluation" if int(value["scientific_seed"]) == 27 else "fitting_observation"
            )
        assignment_records.extend(model_records)
        regrets = [
            value["makespan_regret_seconds"]
            for value in model_records
            if value["strategy"] == "predicted_cost_longest_first"
            and int(value["scientific_seed"]) == 27
        ]
        assignment_regret[
            "historical_spike" if model_name == "historical_spike" else "strongest_non_spike"
        ] = float(np.mean(regrets))
    training_seconds = sum(float(row["client_wall_time_seconds"]) for row in evaluation_seed_rows)
    prediction_fraction = prediction_times["historical_spike"] / training_seconds
    decision = decide_spike_history(
        within_metrics,
        prediction_fraction,
        assignment_regret,
        float(cost_config["rank_correlation_tolerance"]),
        float(cost_config["prediction_time_fraction_limit"]),
        float(cost_config["minimum_runtime_error_improvement_fraction"]),
    )
    decision["strongest_non_spike_name"] = joint_non_spike
    selected_name = "historical_spike" if decision["decision"] == "spike_history_adopted" else joint_non_spike
    ensure_exportable(selected_name)
    scheduler_model = fitted_models[("joint", "client_wall_time_seconds", selected_name)]
    _roundtrip(scheduler_model, root / "client_cost_model.json", evaluation_seed_rows)
    energy_model = fitted_models[("joint", "gross_energy_joules", selected_name)]
    _roundtrip(energy_model, root / "energy_cost_model.json", evaluation_seed_rows)
    evaluation_record = {
        "schema_version": 1,
        "data_separation": {
            "fitting_seeds": [7, 17],
            "evaluation_seed": 27,
            "client_grouped_validation": True,
            "client_id_used_as_predictor": False,
            "prequential_seed_27": True,
            "historical_weight_selection": history_selection,
        },
        "settings": evaluations,
        "spike_history_decision": decision,
        "scheduler_model": selected_name,
        "energy_model": selected_name,
        "all_metrics_finite": all(
            np.isfinite(value)
            for row in metric_rows
            for value in row.values()
            if isinstance(value, (int, float))
        ),
        "model_json_roundtrip_verified": True,
    }
    atomic_write_json(root / "cost_model_evaluation.json", evaluation_record)
    _write_table(root / "cost_model_evaluation.csv", metric_rows)
    markdown = [
        "# Client cost model evaluation\n",
        "\n",
        "This record is generated only after the accepted six-run collection is available.\n",
        "\n",
        f"Spike-history decision: `{decision['decision']}`.\n",
        "\n",
        f"Exported scheduler model: `{selected_name}`. The diagnostic oracle is not exportable.\n",
    ]
    atomic_write_text(root / "cost_model_evaluation.md", "".join(markdown))
    assignment_record = {
        "schema_version": 1,
        "production_scheduler_implemented": False,
        "process_counts": [2, 4],
        "records": assignment_records,
    }
    atomic_write_json(root / "assignment_readiness.json", assignment_record)
    for target, values in prediction_sources.items():
        _write_table(root / f"figure_source_{target}.csv", values)
    _write_table(root / "figure_source_assignment_load.csv", assignment_records)
    return {
        "evaluation": evaluation_record,
        "assignment": assignment_record,
        "rows": rows,
        "selected_scheduler_model": scheduler_model,
        "selected_energy_model": energy_model,
    }


def summarize_resource_measurement(
    run_directories: list[str | Path],
    artifact_root: str | Path,
    result_root: str | Path,
    accounting_path: str | Path,
) -> dict:
    """Require six complete executions and assemble the final descriptive record."""

    artifacts = Path(artifact_root).resolve()
    root = Path(result_root).resolve()
    accounting_source = Path(accounting_path).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not accounting_source.is_file():
        raise FileNotFoundError(accounting_source)
    if len(run_directories) != 6:
        raise ValueError("resource summary requires exactly six run directories")
    run_rows = []
    calibration = []
    summaries = []
    run_provenance = []
    for run_dir_value in sorted(Path(value).resolve() for value in run_directories):
        records, provenance = validate_accepted_run(run_dir_value)
        final = json.loads((run_dir_value / "final_metrics.json").read_text(encoding="utf-8"))
        reference = json.loads((run_dir_value / "calibration_reference.json").read_text(encoding="utf-8"))
        run_rows.extend(records)
        run_provenance.append(provenance)
        calibration.append(reference)
        summaries.append(
            {
                "run_directory": str(run_dir_value),
                "dataset": records[0]["dataset"],
                "scientific_seed": records[0]["scientific_seed"],
                "slurm_allocation_id": provenance["slurm_allocation_id"],
                "slurm_allocation_ids": provenance["slurm_allocation_ids"],
                "client_record_count": len(records),
                "internal_execution_time_seconds": float(final["execution_time_seconds"]),
                "client_wall_time_seconds": sum(value["client_wall_time_seconds"] for value in records),
                "cuda_event_time_seconds": sum(value["cuda_event_time_seconds"] for value in records),
                "gross_energy_joules": sum(value["gross_energy_joules"] for value in records),
                "idle_adjusted_energy_joules": sum(value["idle_adjusted_energy_joules"] for value in records),
            }
        )
    if len(run_rows) != 6000:
        raise ValueError("resource summary requires exactly 6,000 accepted client records")
    matrix = {(value["dataset"], int(value["scientific_seed"])) for value in summaries}
    expected_matrix = {(dataset, seed) for dataset in ("shd", "ssc") for seed in (7, 17, 27)}
    if matrix != expected_matrix:
        raise ValueError("resource summary run matrix is incompatible")
    commits = {value["git_commit"] for value in run_provenance}
    allocations = {
        str(allocation_id)
        for value in run_provenance
        for allocation_id in value["slurm_allocation_ids"]
    }
    if len(commits) != 1 or not allocations:
        raise ValueError("resource summary requires one Git commit and recorded Slurm allocations")
    accounting = _validate_slurm_accounting(accounting_source, allocations)
    for name in (
        "cost_model_evaluation.json",
        "cost_model_evaluation.csv",
        "cost_model_evaluation.md",
        "client_cost_model.json",
        "energy_cost_model.json",
        "assignment_readiness.json",
    ):
        source = artifacts / name
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.resolve() != (root / name).resolve():
            atomic_write_text(root / name, source.read_text(encoding="utf-8"))
    evaluation = json.loads((artifacts / "cost_model_evaluation.json").read_text(encoding="utf-8"))
    if not evaluation.get("all_metrics_finite") or not evaluation.get("model_json_roundtrip_verified"):
        raise ValueError("cost-model evaluation did not pass finite-metric and JSON round-trip gates")
    summary = {
        "schema_version": 1,
        "collection": "resource_measurement",
        "valid": True,
        "completed_run_count": 6,
        "accepted_client_record_count": 6000,
        "datasets": {"shd": 3, "ssc": 3},
        "seeds": [7, 17, 27],
        "calibration_passed": True,
        "power_coverage_complete": True,
        "timing_records_complete": True,
        "energy_integration_complete": True,
        "official_test_access_exactly_once": True,
        "model_json_roundtrip_verified": True,
        "metrics_finite": True,
        "git_commit": next(iter(commits)),
        "slurm_accounting": accounting,
        "resource_totals": {
            "slurm_allocation_elapsed_seconds": accounting["elapsed_seconds"],
            "internal_execution_time_seconds": sum(
                value["internal_execution_time_seconds"] for value in summaries
            ),
            "client_wall_time_seconds": sum(value["client_wall_time_seconds"] for value in summaries),
            "cuda_event_time_seconds": sum(value["cuda_event_time_seconds"] for value in summaries),
            "gross_device_energy_joules": sum(value["gross_energy_joules"] for value in summaries),
            "idle_adjusted_device_energy_joules": sum(
                value["idle_adjusted_energy_joules"] for value in summaries
            ),
            "allocated_gpu_hours": accounting["allocated_gpu_hours"],
            "pending_time_seconds": None,
            "pending_time_included_in_execution_allocation": False,
        },
        "scientific_interpretation": "descriptive_evidence",
        "runs": summaries,
    }
    atomic_write_json(root / "resource_measurement_summary.json", summary)
    _write_table(root / "resource_measurement_summary.csv", summaries)
    lines = [
        "# Resource measurement summary\n",
        "\n",
        "The collection passed its declared consistency gates. Results are descriptive evidence.\n",
        "\n",
        "The spike-history outcome and exported model are recorded in `cost_model_evaluation.md`.\n",
    ]
    atomic_write_text(root / "resource_measurement_summary.md", "".join(lines))
    atomic_write_json(root / "instrumentation_calibration.json", {"schema_version": 1, "records": calibration})
    provenance_root = root / "provenance"
    provenance_root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        provenance_root / "slurm-accounting.txt",
        accounting_source.read_text(encoding="utf-8"),
    )
    atomic_write_json(
        provenance_root / "accounting-sha256.json",
        {"sha256": _file_sha256(accounting_source), "source": str(accounting_source)},
    )
    return summary
