"""Evidence-complete summaries for scaling/energy and non-IID/energy collections."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

import torch

from fedapfa.configuration import (
    COMPARATIVE_SEEDS,
    load_comparative_allocations,
    load_comparative_evaluation_manifest,
)
from fedapfa.configuration.experiment_id import experiment_id
from fedapfa.federated.numerical_equivalence import classify_model_states, prediction_identity
from fedapfa.measurement.records import read_jsonl
from fedapfa.utilities.serialization import atomic_write_json, atomic_write_text

ALLOCATION_RECONCILIATION_TOLERANCE_SECONDS = 2.0


def sample_statistics(values: Sequence[float]) -> dict:
    """Return arithmetic mean and sample standard deviation for exactly three seeds."""

    resolved = [float(value) for value in values]
    if len(resolved) != 3 or any(not math.isfinite(value) for value in resolved):
        raise ValueError("descriptive statistics require exactly three finite seed observations")
    return {
        "mean": statistics.mean(resolved),
        "sample_standard_deviation": statistics.stdev(resolved),
    }


def paired_scaling_metrics(reference_runtime: float, candidate_runtime: float, physical_gpus: int) -> dict:
    if reference_runtime <= 0 or candidate_runtime <= 0 or physical_gpus <= 0:
        raise ValueError("scaling metrics require positive runtimes and physical GPU count")
    speedup = reference_runtime / candidate_runtime
    return {"speedup": speedup, "parallel_efficiency": speedup / physical_gpus}


def _allocated_gpus(value: str) -> int:
    matches = []
    for item in value.split(","):
        key, separator, count = item.partition("=")
        if separator and (key == "gres/gpu" or key.startswith("gres/gpu:")):
            matches.append(int(count))
    if len(matches) != 1 or matches[0] <= 0:
        raise ValueError("Slurm AllocTRES must contain one positive GPU allocation")
    return matches[0]


def _timestamp(value: str, field: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError) as error:
        raise ValueError(f"Slurm {field} timestamp is invalid") from error


def parse_slurm_accounting(path: str | Path) -> tuple[dict[str, dict], str]:
    """Read one allocation row per display array task while retaining raw Slurm IDs."""

    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        rows = list(reader)
        fields = set(reader.fieldnames or ())
    required = {
        "JobID",
        "JobIDRaw",
        "State",
        "ExitCode",
        "ElapsedRaw",
        "AllocTRES",
        "Start",
        "End",
        "NodeList",
    }
    if not rows or not required.issubset(fields):
        raise ValueError(f"Slurm accounting fields are incomplete: {sorted(required - fields)}")
    records = {}
    for row in rows:
        display = row["JobID"]
        raw = row["JobIDRaw"]
        if not display or "." in display or "_" not in display:
            continue
        if display in records:
            raise ValueError(f"Slurm accounting duplicates display array task {display}")
        elapsed = int(row["ElapsedRaw"])
        if elapsed <= 0 or not raw or not row["Start"] or not row["End"] or not row["NodeList"]:
            raise ValueError(f"Slurm accounting row is incomplete for {display}")
        state = row["State"].split()[0].removesuffix("+")
        resumable_states = {"CANCELLED", "PREEMPTED", "REQUEUED", "REVOKED", "TIMEOUT"}
        if state != "COMPLETED" and state not in resumable_states:
            raise ValueError(f"Slurm allocation has a non-resumable failure state: {display}: {row['State']}")
        if state == "COMPLETED" and row["ExitCode"] != "0:0":
            raise ValueError(f"completed Slurm allocation has a nonzero exit code: {display}")
        gpus = _allocated_gpus(row["AllocTRES"])
        records[display] = {
            "display_array_task_id": display,
            "raw_slurm_job_id": raw,
            "state": row["State"],
            "state_base": state,
            "exit_code": row["ExitCode"],
            "resumable_interruption": state in resumable_states,
            "elapsed_seconds": elapsed,
            "allocated_tres": row["AllocTRES"],
            "start": row["Start"],
            "end": row["End"],
            "start_unix_seconds": _timestamp(row["Start"], "start"),
            "end_unix_seconds": _timestamp(row["End"], "end"),
            "node_list": row["NodeList"],
            "physical_gpu_count": gpus,
            "billed_gpu_hours": elapsed * gpus / 3600.0,
        }
    if not records:
        raise ValueError("Slurm accounting contains no display array-task allocation rows")
    return records, digest


def load_allocation_timing(
    path: str | Path,
    *,
    allocation_index: int,
    display_array_task_id: str,
    expected_order: Sequence[str],
) -> dict:
    """Validate complete or interruption-truncated launcher timing for one physical allocation."""

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"launcher allocation timing is missing: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != 1
        or not isinstance(value.get("completed"), bool)
        or value.get("allocation_index") != allocation_index
        or value.get("display_array_task_id") != display_array_task_id
        or value.get("execution_order") != list(expected_order)
        or not value.get("raw_slurm_job_id")
    ):
        raise ValueError(f"launcher allocation timing identity is incompatible: {source}")
    treatments = value.get("treatments")
    open_treatment = value.get("open_treatment")
    if not isinstance(treatments, list) or (open_treatment is not None and not isinstance(open_treatment, Mapping)):
        raise ValueError(f"launcher treatment timing is malformed: {source}")
    observed = [*treatments, *([] if open_treatment is None else [open_treatment])]
    observed_ids = [item.get("treatment_id") for item in observed]
    if observed_ids != list(expected_order[: len(observed)]):
        raise ValueError(f"launcher treatment timing order is incompatible: {source}")
    if value["completed"]:
        if (
            open_treatment is not None
            or observed_ids != list(expected_order)
            or not isinstance(value.get("launcher_observed_allocation_end"), Mapping)
        ):
            raise ValueError(f"completed launcher timing is incomplete: {source}")
    elif value.get("launcher_observed_allocation_end") is not None:
        raise ValueError(f"interrupted launcher timing has a completion boundary: {source}")
    try:
        boundaries = [int(value["launcher_observed_allocation_start"]["monotonic_timestamp_ns"])]
        for position, item in enumerate(observed, start=1):
            if item.get("treatment_position") != position or not isinstance(item.get("start"), Mapping):
                raise ValueError
            boundaries.append(int(item["start"]["monotonic_timestamp_ns"]))
            if item.get("end") is not None:
                boundaries.append(int(item["end"]["monotonic_timestamp_ns"]))
                if float(item.get("launcher_observed_duration_seconds", 0.0)) <= 0:
                    raise ValueError
            elif item is not open_treatment or item.get("launcher_observed_duration_seconds") is not None:
                raise ValueError
        if value["completed"]:
            boundaries.append(int(value["launcher_observed_allocation_end"]["monotonic_timestamp_ns"]))
            if float(value.get("launcher_observed_allocation_duration_seconds", 0.0)) <= 0:
                raise ValueError
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"launcher allocation timing boundaries are malformed: {source}") from error
    if any(item < 0 for item in boundaries) or any(
        right <= left for left, right in zip(boundaries, boundaries[1:], strict=False)
    ):
        raise ValueError(f"launcher allocation timing boundaries are nonmonotonic: {source}")
    return value


def reconcile_allocation(
    accounting: Mapping,
    runs: Sequence[dict],
    *,
    expected_order: Sequence[str],
    launcher_timing: Mapping | None = None,
) -> dict:
    """Reconcile one physical allocation without duplicating billing across treatments."""

    ordered = sorted(runs, key=lambda value: int(value["treatment_position"]))
    if [value["treatment_id"] for value in ordered] != list(expected_order):
        raise ValueError("allocation treatment order differs from the manifest")
    if len({(value["dataset"], value["seed"]) for value in ordered}) != 1:
        raise ValueError("allocation crosses dataset or seed identities")
    elapsed = float(accounting["elapsed_seconds"])
    if launcher_timing is None:
        allocation_start = float(accounting["start_unix_seconds"])
        allocation_end = float(accounting["end_unix_seconds"])
        initialization = float(ordered[0]["execution_start_unix_seconds"]) - allocation_start
        between = [
            float(following["execution_start_unix_seconds"]) - float(previous["execution_end_unix_seconds"])
            for previous, following in zip(ordered, ordered[1:], strict=False)
        ]
        remaining = allocation_end - float(ordered[-1]["execution_end_unix_seconds"])
        if any(value < -ALLOCATION_RECONCILIATION_TOLERANCE_SECONDS for value in (initialization, *between, remaining)):
            raise ValueError("allocation contains overlapping or out-of-bound treatment intervals")
        initialization = max(initialization, 0.0)
        between = [max(value, 0.0) for value in between]
        remaining = max(remaining, 0.0)
        treatment_seconds = sum(float(value["runtime_seconds"]) for value in ordered)
        treatment_records = [
            {
                "treatment_id": value["treatment_id"],
                "treatment_position": value["treatment_position"],
                "launcher_observed_treatment_duration_seconds": value["runtime_seconds"],
                "derived_physical_allocation_treatment_gpu_exposure_hours": value["derived_gpu_exposure_hours"],
                "timing_status": "complete",
            }
            for value in ordered
        ]
        launcher_completed = True
    else:
        start_ns = int(launcher_timing["launcher_observed_allocation_start"]["monotonic_timestamp_ns"])
        closed = list(launcher_timing["treatments"])
        open_treatment = launcher_timing.get("open_treatment")
        observed = [*closed, *([] if open_treatment is None else [open_treatment])]
        initialization = (
            (int(observed[0]["start"]["monotonic_timestamp_ns"]) - start_ns) / 1_000_000_000 if observed else 0.0
        )
        between = []
        for previous, following in zip(observed, observed[1:], strict=False):
            if previous.get("end") is None:
                break
            between.append(
                (int(following["start"]["monotonic_timestamp_ns"]) - int(previous["end"]["monotonic_timestamp_ns"]))
                / 1_000_000_000
            )
        treatment_seconds = sum(float(value["launcher_observed_duration_seconds"]) for value in closed)
        known_seconds = initialization + sum(between) + treatment_seconds
        remaining = max(elapsed - known_seconds, 0.0)
        launcher_completed = bool(launcher_timing["completed"])
        timing_by_id = {value["treatment_id"]: value for value in observed}
        treatment_records = []
        for position, treatment_id in enumerate(expected_order, start=1):
            timing = timing_by_id.get(treatment_id)
            duration = (
                None
                if timing is None or timing.get("end") is None
                else float(timing["launcher_observed_duration_seconds"])
            )
            treatment_records.append(
                {
                    "treatment_id": treatment_id,
                    "treatment_position": position,
                    "launcher_observed_treatment_duration_seconds": duration,
                    "derived_physical_allocation_treatment_gpu_exposure_hours": (
                        None if duration is None else duration * int(accounting["physical_gpu_count"]) / 3600.0
                    ),
                    "timing_status": (
                        "not_started"
                        if timing is None
                        else "interrupted_not_closed"
                        if timing.get("end") is None
                        else "complete"
                    ),
                }
            )
    if any(value < 0 for value in (initialization, *between, remaining)):
        raise ValueError("allocation timing contains negative components")
    reconciliation_error = elapsed - (treatment_seconds + initialization + sum(between) + remaining)
    if abs(reconciliation_error) > ALLOCATION_RECONCILIATION_TOLERANCE_SECONDS:
        raise ValueError(
            f"allocation reconciliation error exceeds {ALLOCATION_RECONCILIATION_TOLERANCE_SECONDS} seconds"
        )
    return {
        **dict(accounting),
        "dataset": ordered[0]["dataset"],
        "seed": ordered[0]["seed"],
        "treatment_order": list(expected_order),
        "allocation_initialization_seconds": initialization,
        "between_treatment_seconds": sum(between),
        "between_treatment_intervals_seconds": between,
        "remaining_allocation_overhead_seconds": remaining,
        "interrupted_unclassified_seconds": 0.0 if launcher_completed else remaining,
        "allocation_reconciliation_error_seconds": reconciliation_error,
        "allocation_reconciliation_tolerance_seconds": ALLOCATION_RECONCILIATION_TOLERANCE_SECONDS,
        "treatments": treatment_records,
        "billing_scope": "one physical Slurm allocation; never duplicated across treatments",
        "derived_treatment_gpu_exposure_billed_separately": False,
        "launcher_timing_completed": launcher_completed,
        "launcher_monotonic_timing": None if launcher_timing is None else dict(launcher_timing),
    }


def classify_numerical_evidence(
    *,
    reference_state: Mapping[str, torch.Tensor],
    candidate_state: Mapping[str, torch.Tensor],
    reference_selected_clients: Sequence,
    candidate_selected_clients: Sequence,
    reference_client_seeds: Sequence,
    candidate_client_seeds: Sequence,
    reference_ordered_updates: Sequence,
    candidate_ordered_updates: Sequence,
    reference_weights: Sequence,
    candidate_weights: Sequence,
    reference_checkpoint_round: int,
    candidate_checkpoint_round: int,
    reference_predictions: list[int] | None,
    candidate_predictions: list[int] | None,
    reference_metrics: Mapping,
    candidate_metrics: Mapping,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict:
    """Classify structure, exactness, bounds, predictions, checkpoints, and metrics separately."""

    parameters = classify_model_states(
        reference_state,
        candidate_state,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    selected_identity = list(reference_selected_clients) == list(candidate_selected_clients)
    seed_identity = list(reference_client_seeds) == list(candidate_client_seeds)
    update_identity = list(reference_ordered_updates) == list(candidate_ordered_updates)
    weight_identity = list(reference_weights) == list(candidate_weights)
    structural = selected_identity and seed_identity and update_identity and weight_identity
    prediction = prediction_identity(reference_predictions, candidate_predictions)
    checkpoint_round_identity = reference_checkpoint_round == candidate_checkpoint_round
    metric_identity = dict(reference_metrics) == dict(candidate_metrics)
    return {
        "structural_identity": structural,
        "selected_client_identity": selected_identity,
        "client_seed_identity": seed_identity,
        "ordered_client_update_identity": update_identity,
        "aggregation_weight_identity": weight_identity,
        "exact_parameter_identity": parameters.bitwise_parameter_identity,
        "bounded_parameter_identity": parameters.mathematical_equivalence,
        "maximum_absolute_parameter_difference": parameters.maximum_absolute_parameter_difference,
        "maximum_relative_parameter_difference": parameters.maximum_relative_parameter_difference,
        "finite_parameters": parameters.finite_parameters,
        "prediction_identity": prediction,
        "checkpoint_round_identity": checkpoint_round_identity,
        "checkpoint_identity": checkpoint_round_identity and parameters.bitwise_parameter_identity,
        "metric_identity": metric_identity,
        "exact_equivalence": (
            structural
            and parameters.bitwise_parameter_identity
            and prediction is True
            and checkpoint_round_identity
            and metric_identity
        ),
    }


def _state(run_dir: Path, final: Mapping) -> dict[str, torch.Tensor]:
    selected = final.get("selected_checkpoint_artifact")
    if not isinstance(selected, str) or not selected:
        raise ValueError("selected checkpoint artifact is missing")
    checkpoint = torch.load(run_dir / selected, map_location="cpu", weights_only=False)
    state = checkpoint.get("global_model_state")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("selected checkpoint has no global model state")
    return {name: value.detach().cpu() for name, value in state.items()}


def _imbalance(values: Sequence[float]) -> float:
    resolved = [float(value) for value in values]
    if not resolved or max(resolved) <= 0:
        return 0.0
    return (max(resolved) - min(resolved)) / max(resolved)


def _gpu_utilization(run_dir: Path, energy: Mapping) -> float:
    attempts = energy.get("attempts", [])
    completed = next((value for value in reversed(attempts) if value.get("execution_completed")), None)
    if completed is None:
        raise ValueError("completed energy attempt is missing")
    attempt = int(completed["execution_attempt"])
    rows = read_jsonl(run_dir / "measurement_attempts" / f"attempt_{attempt}" / "merged_device_samples.jsonl")
    values = [
        float(value["gpu_utilization_percent"])
        for value in rows
        if value.get("sampling_error_status") is None and value.get("gpu_utilization_percent") is not None
    ]
    if not values:
        raise ValueError("GPU utilization samples are unavailable")
    return statistics.mean(values)


def _run_record(task, runs_root: Path, allocation_lookup: Mapping[tuple[str, int, str], tuple[int, int]]) -> dict:
    config = task.config
    run_dir = runs_root / experiment_id(config)
    required = {
        "final": run_dir / "final_metrics.json",
        "acceptance": run_dir / "acceptance.json",
        "measurement": run_dir / "measurement_acceptance.json",
        "energy": run_dir / "energy_summary.json",
        "measurements": run_dir / "execution_measurements.json",
        "partition": run_dir / "partition.json",
        "official": run_dir / "official_test_metrics.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise ValueError(f"{task.experiment} seed {task.seed} lacks evidence: {missing}")
    values = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in required.items()}
    final = values["final"]
    measurement = values["measurement"]
    energy = values["energy"]
    execution = values["measurements"]
    official = values["official"]
    if not final.get("completed") or not values["acceptance"].get("completed"):
        raise ValueError(f"{task.experiment} seed {task.seed} execution is incomplete")
    if not measurement.get("accepted") or not energy.get("execution_completed"):
        raise ValueError(f"{task.experiment} seed {task.seed} energy evidence is incomplete")
    if official.get("access_count") != 1 or not official.get("evaluation_completed"):
        raise ValueError(f"{task.experiment} seed {task.seed} official-test evidence is invalid")
    treatment = config["comparative_evaluation"]["treatment_id"]
    allocation_index, position = allocation_lookup[(task.dataset, task.seed, treatment)]
    complete_energy = energy["complete_treatment_energy"]
    client_energy = energy["accepted_client_training_energy"]
    rounds = read_jsonl(run_dir / "round_metrics.jsonl")
    clients = read_jsonl(run_dir / "client_metrics.jsonl")
    resource_rows = read_jsonl(run_dir / "client_resource_records.jsonl")
    if len(rounds) != 100 or len(clients) != 1000 or len(resource_rows) != 1000:
        raise ValueError(f"{task.experiment} seed {task.seed} record counts are incomplete")
    partition_clients = values["partition"]["clients"]
    populations = [float(value["example_count"]) for value in partition_clients]
    event_totals = defaultdict(float)
    for value in resource_rows:
        event_totals[str(value["client_id"])] += float(value["total_raw_input_events"])
    runtime = float(execution["internal_treatment_duration_seconds"])
    device_count = int(config["parallel_execution"]["device_count"])
    test = official["metrics"]
    transfer = {
        "client_wall_time": {
            "mean_absolute_error": statistics.mean(
                abs(
                    float(value["pre_execution_predicted_client_wall_time_seconds"])
                    - float(value["client_wall_time_seconds"])
                )
                for value in resource_rows
            )
        },
        "gross_energy": {
            "mean_absolute_error": statistics.mean(
                abs(float(value["pre_execution_predicted_gross_energy_joules"]) - float(value["gross_energy_joules"]))
                for value in resource_rows
            )
        },
        "idle_adjusted_energy": {
            "available": False,
            "mean_absolute_error": None,
            "reason": "no compatible frozen Week 5 idle-adjusted-energy target artifact",
        },
        "cuda_event_time": {
            "available": False,
            "mean_absolute_error": None,
            "reason": "no compatible frozen Week 5 CUDA-event-time target artifact",
        },
        "classification": "frozen_transfer_evaluation",
        "models_refit_with_current_observations": False,
    }
    result = {
        "experiment": task.experiment,
        "dataset": task.dataset,
        "seed": task.seed,
        "treatment_id": treatment,
        "allocation_index": allocation_index,
        "treatment_position": position,
        "run_directory": str(run_dir),
        "selected_round": int(final["selected_round"]),
        "validation_metric": final["selected_validation"]["accuracy"],
        "official_test_accuracy": float(test["accuracy"]),
        "macro_f1": float(test["macro_f1"]),
        "runtime_seconds": runtime,
        "mean_round_time_seconds": statistics.mean(float(value["total_round_time_seconds"]) for value in rounds),
        "client_wall_time_seconds": sum(float(value["training_duration_seconds"]) for value in clients),
        "aggregation_time_seconds": sum(float(value["aggregation_time_seconds"]) for value in rounds),
        "validation_time_seconds": sum(float(value["validation_time_seconds"]) for value in rounds),
        "official_test_time_seconds": float(execution["official_test_time_seconds"]),
        "gross_execution_energy_joules": float(complete_energy["gross_energy_joules"]),
        "idle_adjusted_execution_energy_joules": float(complete_energy["idle_adjusted_energy_joules"]),
        "client_training_energy_joules": float(client_energy["gross_energy_joules"]),
        "physical_gpu_count": device_count,
        "derived_gpu_exposure_hours": runtime * device_count / 3600.0,
        "gpu_utilization_percent": _gpu_utilization(run_dir, energy),
        "peak_allocated_cuda_memory_bytes": max(
            int(value.get("peak_allocated_cuda_memory_bytes") or 0) for value in resource_rows
        ),
        "peak_reserved_cuda_memory_bytes": max(
            int(value.get("peak_reserved_cuda_memory_bytes") or 0) for value in resource_rows
        ),
        "load_imbalance": statistics.mean(float(value["process_load_imbalance"]) for value in execution["rounds"]),
        "client_population_imbalance": _imbalance(populations),
        "client_event_imbalance": _imbalance(event_totals.values()),
        "logical_communication_bytes": int(final["logical_communication"]["cumulative_total_bytes"]),
        "logical_inter_node_movement_bytes": sum(
            int(value["logical_inter_node_bytes"]) for value in execution["rounds"]
        ),
        "execution_start_unix_seconds": float(execution["execution_start_unix_seconds"]),
        "execution_end_unix_seconds": float(execution["execution_end_unix_seconds"]),
        "resource_allocation": execution["resource_allocation"],
        "resource_allocations": execution.get("resource_allocations", [execution["resource_allocation"]]),
        "partition_id": final["partition_id"],
        "partition_alpha": final["partition_alpha"],
        "partition_method": final["distribution"],
        "partition_construction_attempts": values["partition"].get("construction_attempts", 1),
        "partition_retries": values["partition"].get("construction_attempts", 1) - 1,
        "partition_repair_actions": [],
        "resolved_partition_seed": values["partition"]["partition_seed"],
        "total_eligible_training_examples": values["partition"]["eligible_training_examples"],
        "client_populations": populations,
        "client_presented_examples": {
            str(client["client_id"]): sum(
                int(value["local_training_examples_presented"])
                for value in clients
                if str(value["client_id"]) == str(client["client_id"])
            )
            for client in partition_clients
        },
        "class_counts": {value["client_id"]: value["class_counts"] for value in partition_clients},
        "represented_class_counts": {value["client_id"]: value["represented_labels"] for value in partition_clients},
        "label_entropy_bits": {value["client_id"]: value["label_entropy_bits"] for value in partition_clients},
        "partition_integrity": values["partition"]["integrity_checks"],
        "round_records": rounds,
        "client_records": clients,
        "official_predictions": test.get("predictions"),
        "official_metrics": {"accuracy": test["accuracy"], "macro_f1": test["macro_f1"]},
        "state": _state(run_dir, final),
        "frozen_model_transfer_evaluation": transfer,
        "frozen_model_provenance": config["frozen_model_diagnostics"],
        "configuration_whitelist_version": config["comparative_evaluation"]["resolved_configuration_whitelist_version"],
    }
    if any(not value for value in result["partition_integrity"].values()):
        raise ValueError(f"{task.experiment} seed {task.seed} partition integrity failed")
    return result


def _allocation_lookup(allocations) -> dict[tuple[str, int, str], tuple[int, int]]:
    return {
        (allocation.dataset, allocation.seed, treatment): (allocation.allocation_index, position)
        for allocation in allocations
        for position, treatment in enumerate(allocation.execution_order, start=1)
    }


def _trajectory(run: Mapping) -> tuple[list, list, list, list]:
    rounds = run["round_records"]
    clients = sorted(
        run["client_records"], key=lambda value: (int(value["round_number"]), int(value["selected_position"]))
    )
    return (
        [value["selected_client_ids"] for value in rounds],
        [int(value["resolved_training_seed"]) for value in clients],
        [value["ordered_update_identities"] for value in rounds],
        [value["aggregation_weights"] for value in rounds],
    )


def summarize_comparative_evaluation(
    manifest_path: str | Path,
    runs_root: str | Path,
    output_dir: str | Path,
    *,
    slurm_accounting: str | Path,
) -> dict:
    """Validate all executions and write deterministic JSON, CSV, and Markdown summaries."""

    tasks = load_comparative_evaluation_manifest(manifest_path)
    allocations = load_comparative_allocations(manifest_path)
    collection = tasks[0].config["comparative_evaluation"]["collection"]
    resolved_runs_root = Path(runs_root)
    runs = [_run_record(task, resolved_runs_root, _allocation_lookup(allocations)) for task in tasks]
    accounting, accounting_hash = parse_slurm_accounting(slurm_accounting)
    timing_root = resolved_runs_root / "allocation_timing"
    timing_paths = sorted(timing_root.glob("*.json"))
    timing_identities = {}
    for timing_path in timing_paths:
        timing_identity = json.loads(timing_path.read_text(encoding="utf-8"))
        display = timing_identity.get("display_array_task_id")
        if not isinstance(display, str) or not display or display in timing_identities:
            raise ValueError(f"launcher allocation timing identity is duplicate or missing: {timing_path}")
        timing_identities[display] = (timing_path, timing_identity)
    if set(accounting) != set(timing_identities):
        raise ValueError("Slurm accounting and launcher timing cover different physical allocations")

    recorded_attempt_allocations = set()
    for run in runs:
        histories = run.get("resource_allocations")
        if not isinstance(histories, list) or not histories:
            raise ValueError(f"{run['experiment']} seed {run['seed']} allocation history is missing")
        for resource in histories:
            if not isinstance(resource, Mapping):
                raise ValueError("execution allocation history is malformed")
            display = f"{resource.get('array_job_id')}_{resource.get('array_task_id')}"
            if display not in timing_identities or display not in accounting:
                raise ValueError(f"execution allocation history lacks timing/accounting evidence: {display}")
            if str(accounting[display]["raw_slurm_job_id"]) != str(resource.get("job_id")):
                raise ValueError(f"Slurm raw job identity differs for {display}")
            recorded_attempt_allocations.add(display)

    reconciled = []
    billed_by_index = defaultdict(float)
    for allocation in allocations:
        allocation_runs = [value for value in runs if value["allocation_index"] == allocation.allocation_index]
        selected_timings = [
            (display, path, identity)
            for display, (path, identity) in timing_identities.items()
            if identity.get("allocation_index") == allocation.allocation_index
        ]
        if not selected_timings:
            raise ValueError(f"logical allocation {allocation.allocation_index} has no physical allocation evidence")
        completed_physical_allocations = 0
        for display, timing_path, timing_identity in sorted(selected_timings):
            record = accounting[display]
            if int(record["physical_gpu_count"]) != int(allocation_runs[0]["physical_gpu_count"]):
                raise ValueError(f"Slurm accounting GPU count differs for {display}")
            if str(record["raw_slurm_job_id"]) != str(timing_identity.get("raw_slurm_job_id")):
                raise ValueError(f"launcher and Slurm raw job identities differ for {display}")
            launcher_timing = load_allocation_timing(
                timing_path,
                allocation_index=allocation.allocation_index,
                display_array_task_id=display,
                expected_order=allocation.execution_order,
            )
            physical = reconcile_allocation(
                record,
                allocation_runs,
                expected_order=allocation.execution_order,
                launcher_timing=launcher_timing,
            )
            physical["logical_allocation_index"] = allocation.allocation_index
            reconciled.append(physical)
            billed_by_index[allocation.allocation_index] += float(record["billed_gpu_hours"])
            if record["state_base"] == "COMPLETED" and launcher_timing["completed"]:
                completed_physical_allocations += 1
        if completed_physical_allocations != 1:
            raise ValueError(
                f"logical allocation {allocation.allocation_index} requires exactly one "
                "successful completion allocation"
            )
    if not recorded_attempt_allocations.issubset(timing_identities):
        raise ValueError("an execution attempt allocation lacks launcher timing")
    if collection == "system_scaling_energy_evaluation":
        for run in runs:
            run["allocated_gpu_hours"] = billed_by_index[run["allocation_index"]]
    numerical = []
    paired = []
    if collection == "system_scaling_energy_evaluation":
        for dataset in ("shd", "ssc"):
            for seed in COMPARATIVE_SEEDS:
                reference = next(
                    value
                    for value in runs
                    if value["dataset"] == dataset
                    and value["seed"] == seed
                    and value["treatment_id"] == "one_node_one_gpu"
                )
                reference_trajectory = _trajectory(reference)
                for candidate in [value for value in runs if value["dataset"] == dataset and value["seed"] == seed]:
                    candidate_trajectory = _trajectory(candidate)
                    classification = classify_numerical_evidence(
                        reference_state=reference["state"],
                        candidate_state=candidate["state"],
                        reference_selected_clients=reference_trajectory[0],
                        candidate_selected_clients=candidate_trajectory[0],
                        reference_client_seeds=reference_trajectory[1],
                        candidate_client_seeds=candidate_trajectory[1],
                        reference_ordered_updates=reference_trajectory[2],
                        candidate_ordered_updates=candidate_trajectory[2],
                        reference_weights=reference_trajectory[3],
                        candidate_weights=candidate_trajectory[3],
                        reference_checkpoint_round=reference["selected_round"],
                        candidate_checkpoint_round=candidate["selected_round"],
                        reference_predictions=reference["official_predictions"],
                        candidate_predictions=candidate["official_predictions"],
                        reference_metrics=reference["official_metrics"],
                        candidate_metrics=candidate["official_metrics"],
                        absolute_tolerance=float(candidate["round_records"][0].get("absolute_tolerance", 1e-6)),
                        relative_tolerance=1e-5,
                    )
                    classification.update({"dataset": dataset, "seed": seed, "treatment_id": candidate["treatment_id"]})
                    numerical.append(classification)
                    scaling = paired_scaling_metrics(
                        reference["runtime_seconds"],
                        candidate["runtime_seconds"],
                        candidate["physical_gpu_count"],
                    )
                    candidate.update(scaling)
                    candidate["energy_ratio_relative_to_one_gpu"] = (
                        candidate["gross_execution_energy_joules"] / reference["gross_execution_energy_joules"]
                    )
                    candidate["energy_delay_product_joule_seconds"] = (
                        candidate["gross_execution_energy_joules"] * candidate["runtime_seconds"]
                    )
                    candidate["numerical_identity"] = classification
                    paired.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "treatment_id": candidate["treatment_id"],
                            **scaling,
                            "energy_ratio_relative_to_one_gpu": candidate["energy_ratio_relative_to_one_gpu"],
                        }
                    )
    else:
        for dataset in ("shd", "ssc"):
            for seed in COMPARATIVE_SEEDS:
                reference = next(
                    value
                    for value in runs
                    if value["dataset"] == dataset and value["seed"] == seed and value["treatment_id"] == "iid"
                )
                for candidate in [value for value in runs if value["dataset"] == dataset and value["seed"] == seed]:
                    candidate["accuracy_difference_from_paired_iid"] = (
                        candidate["official_test_accuracy"] - reference["official_test_accuracy"]
                    )
                    candidate["macro_f1_difference_from_paired_iid"] = candidate["macro_f1"] - reference["macro_f1"]
                    candidate["energy_per_communication_round_joules"] = (
                        candidate["gross_execution_energy_joules"] / 100
                    )
                    candidate["energy_per_accepted_client_joules"] = candidate["client_training_energy_joules"] / 1000
                    paired.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "treatment_id": candidate["treatment_id"],
                            "accuracy_difference_from_paired_iid": candidate["accuracy_difference_from_paired_iid"],
                            "macro_f1_difference_from_paired_iid": candidate["macro_f1_difference_from_paired_iid"],
                        }
                    )
    metric_fields = (
        "official_test_accuracy",
        "macro_f1",
        "selected_round",
        "runtime_seconds",
        "mean_round_time_seconds",
        "client_wall_time_seconds",
        "aggregation_time_seconds",
        "validation_time_seconds",
        "official_test_time_seconds",
        "gross_execution_energy_joules",
        "idle_adjusted_execution_energy_joules",
        "client_training_energy_joules",
        "derived_gpu_exposure_hours",
        "gpu_utilization_percent",
        "peak_allocated_cuda_memory_bytes",
        "load_imbalance",
        "logical_communication_bytes",
        "logical_inter_node_movement_bytes",
    )
    if collection == "system_scaling_energy_evaluation":
        metric_fields += (
            "allocated_gpu_hours",
            "speedup",
            "parallel_efficiency",
            "energy_ratio_relative_to_one_gpu",
            "energy_delay_product_joule_seconds",
        )
    else:
        metric_fields += (
            "validation_metric",
            "accuracy_difference_from_paired_iid",
            "macro_f1_difference_from_paired_iid",
            "client_population_imbalance",
            "client_event_imbalance",
            "energy_per_communication_round_joules",
            "energy_per_accepted_client_joules",
        )
    groups = []
    for dataset in ("shd", "ssc"):
        for treatment in dict.fromkeys(value["treatment_id"] for value in runs if value["dataset"] == dataset):
            selected = [value for value in runs if value["dataset"] == dataset and value["treatment_id"] == treatment]
            if sorted(value["seed"] for value in selected) != list(COMPARATIVE_SEEDS):
                raise ValueError("summary grouping is missing a dataset/seed pair")
            groups.append(
                {
                    "dataset": dataset,
                    "treatment_id": treatment,
                    "completed_seeds": list(COMPARATIVE_SEEDS),
                    "statistics": {
                        field: sample_statistics([float(value[field]) for value in selected]) for field in metric_fields
                    },
                }
            )
    for value in runs:
        value.pop("state", None)
        value.pop("round_records", None)
        value.pop("client_records", None)
        value.pop("official_predictions", None)
    outcome = (
        "system_scaling_energy_characterization_complete"
        if collection == "system_scaling_energy_evaluation"
        else "non_iid_energy_characterization_complete"
    )
    summary = {
        "schema_version": 1,
        "collection": collection,
        "valid": True,
        "outcome": outcome,
        "task_count": 24,
        "allocation_count": len(allocations),
        "logical_allocation_count": len(allocations),
        "physical_allocation_count": len(reconciled),
        "resumable_interruption_allocation_count": sum(value["resumable_interruption"] for value in reconciled),
        "seeds": list(COMPARATIVE_SEEDS),
        "datasets_pooled": False,
        "statistical_scope": "descriptive arithmetic mean and sample standard deviation; no significance test",
        "negative_or_nonmonotonic_outcome_accepted": True,
        "logical_communication_is_measured_physical_traffic": False,
        "runs": runs,
        "paired_records": paired,
        "numerical_identity": numerical,
        "groups": groups,
        "allocations": reconciled,
        "total_billed_slurm_gpu_hours": sum(value["billed_gpu_hours"] for value in reconciled),
        "slurm_accounting_sha256": accounting_hash,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = collection
    atomic_write_json(output / f"{stem}_summary.json", summary)
    scalar_fields = sorted(
        key
        for key in set().union(*(value.keys() for value in runs))
        if all(not isinstance(value.get(key), (dict, list)) for value in runs)
    )
    lines = []
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=scalar_fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(runs)
    atomic_write_text(output / f"{stem}_summary.csv", stream.getvalue())
    lines.extend(
        (
            f"# {collection.replace('_', ' ').title()}",
            "",
            f"Outcome: `{outcome}`.",
            "",
            "This summary is descriptive, keeps SHD and SSC separate, and contains no significance test.",
            "",
            f"Completed executions: {len(runs)} in {len(reconciled)} Slurm allocations.",
            "",
        )
    )
    atomic_write_text(output / f"{stem}_summary.md", "\n".join(lines))
    return summary
