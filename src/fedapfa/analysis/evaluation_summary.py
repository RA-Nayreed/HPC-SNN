"""Strict paired summaries for scheduling and hierarchical-reduction evidence."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import statistics
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch

from fedapfa.configuration import (
    EVALUATION_SEEDS,
    load_evaluation_manifest,
    validate_resolved_evaluation_pair,
)
from fedapfa.distributed.process_context import allocated_gpu_uuids, canonical_gpu_uuid
from fedapfa.federated.aggregation import aggregation_tensor_policy
from fedapfa.federated.checkpointing import configuration_identity
from fedapfa.federated.numerical_equivalence import classify_model_states, prediction_identity
from fedapfa.scheduling.base import EVENT_STRUCTURE_FEATURES
from fedapfa.utilities.run_records import run_directory

ALLOCATION_RECONCILIATION_TOLERANCE_SECONDS = 2.0


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _jsonl(path: Path) -> list[dict]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(value, dict) for value in values):
        raise ValueError(f"{path} contains a non-object record")
    return values


def _finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _nonnegative(value, name: str) -> float:
    result = _finite(value, name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive(value, name: str) -> float:
    result = _finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _fraction(value, name: str) -> float:
    result = _finite(value, name)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be between zero and one")
    return result


def _allocated_gpus(value: str) -> int:
    candidates = []
    for item in value.split(","):
        key, separator, count = item.partition("=")
        if separator and (key == "gres/gpu" or key.startswith("gres/gpu:")):
            candidates.append(int(count))
    if len(candidates) != 1 or candidates[0] <= 0:
        raise ValueError("Slurm AllocTRES must contain one positive GPU allocation")
    return candidates[0]


def _timestamp(value: str, label: str) -> float:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.timestamp()
    except (TypeError, ValueError) as error:
        raise ValueError(f"Slurm {label} timestamp is invalid") from error


def _slurm_accounting(path: str | Path) -> tuple[dict[str, dict], str, str]:
    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        rows = list(reader)
        fields = set(reader.fieldnames or ())
    identifier_fields = fields.intersection({"JobID", "JobIDRaw"})
    if len(identifier_fields) != 1:
        raise ValueError("Slurm accounting must contain exactly one of JobID or JobIDRaw")
    identifier_field = identifier_fields.pop()
    required = {
        identifier_field,
        "State",
        "ExitCode",
        "ElapsedRaw",
        "AllocTRES",
        "Start",
        "End",
    }
    if not rows or any(not required.issubset(row) for row in rows):
        raise ValueError("Slurm accounting fields are incomplete")
    records = {}
    for row in rows:
        job_id = row[identifier_field]
        if not job_id or "." in job_id or "_" not in job_id:
            continue
        if job_id in records:
            raise ValueError(f"Slurm accounting contains duplicate array task {job_id}")
        elapsed = int(row["ElapsedRaw"])
        if elapsed <= 0 or not row["Start"] or not row["End"]:
            raise ValueError(f"Slurm accounting timing is incomplete for {job_id}")
        gpu_count = _allocated_gpus(row["AllocTRES"])
        records[job_id] = {
            "job_id": job_id,
            "state": row["State"],
            "exit_code": row["ExitCode"],
            "slurm_allocation_elapsed_seconds": elapsed,
            "allocated_gpu_count": gpu_count,
            "slurm_allocation_gpu_hours": elapsed * gpu_count / 3600.0,
            "allocation": row["AllocTRES"],
            "start": row["Start"],
            "end": row["End"],
            "start_unix_seconds": _timestamp(row["Start"], "start"),
            "end_unix_seconds": _timestamp(row["End"], "end"),
        }
    if not records:
        raise ValueError(
            "Slurm accounting contains no array-task records; use JobID when JobIDRaw omits array indices"
        )
    return records, digest, identifier_field


def _materialize_runtime_data_root(tasks: list, data_root: str | Path | None) -> list:
    if data_root is None:
        return tasks
    root = Path(data_root)
    if not root.is_absolute():
        raise ValueError("runtime data root must be an absolute path")
    materialized = []
    for task in tasks:
        config = copy.deepcopy(task.config)
        config["dataset"]["root"] = str(root / task.dataset)
        materialized.append(replace(task, config=config))
    return materialized


def _state(path: Path, final: dict) -> dict[str, torch.Tensor]:
    selected = final.get("selected_checkpoint_artifact")
    if not isinstance(selected, str) or not selected:
        raise ValueError("selected checkpoint artifact is missing")
    checkpoint = torch.load(path / selected, map_location="cpu", weights_only=False)
    state = checkpoint.get("global_model_state")
    if not isinstance(state, dict) or not state:
        raise ValueError("selected checkpoint state is missing")
    return {name: tensor.detach().cpu() for name, tensor in state.items()}


def _task_path(root: Path, collection: str, config: dict) -> Path:
    relative = run_directory(config)
    if root.name == collection:
        return root / relative.name
    if root.name == "runs":
        return root / collection / relative.name
    return root / relative.name


def _expected_treatment_order(collection: str, seed: int) -> tuple[str, ...]:
    if collection == "scheduling_evaluation":
        return {
            37: (
                "round_robin",
                "example_count_longest_processing_time",
                "event_structure_longest_processing_time",
            ),
            47: (
                "example_count_longest_processing_time",
                "event_structure_longest_processing_time",
                "round_robin",
            ),
            57: (
                "event_structure_longest_processing_time",
                "round_robin",
                "example_count_longest_processing_time",
            ),
        }[seed]
    return {
        37: ("flat_ordered", "node_hierarchical"),
        47: ("node_hierarchical", "flat_ordered"),
        57: ("flat_ordered", "node_hierarchical"),
    }[seed]


def _load_run(task, root: Path, collection: str) -> dict:
    path = _task_path(root, collection, task.config)
    acceptance = _json(path / "acceptance.json")
    final = _json(path / "final_metrics.json")
    official = _json(path / "official_test_metrics.json")
    measurements = _json(path / "execution_measurements.json")
    rounds = _jsonl(path / "round_metrics.jsonl")
    clients = _jsonl(path / "client_metrics.jsonl")
    environment = _json(path / "environment.json")
    git_record = _json(path / "git.json")
    configured_rounds = task.config["federated"]["rounds"]
    expected_clients = configured_rounds * task.config["federated"]["clients_per_round"]
    if acceptance.get("completed") is not True or final.get("completed") is not True:
        raise ValueError("execution completion did not pass")
    if len(rounds) != configured_rounds or len(clients) != expected_clients:
        raise ValueError("round or client records are incomplete")
    if measurements.get("completed") is not True or len(measurements.get("rounds", [])) != configured_rounds:
        raise ValueError("execution measurements are incomplete")
    if official.get("access_count") != 1 or official.get("evaluation_completed") is not True:
        raise ValueError("official test isolation is incomplete")
    if final.get("scheduler_model", {}).get("model_load_count") != 1:
        raise ValueError("frozen scheduler model was not loaded exactly once")
    if (
        final.get("scheduler_model", {}).get("model_sha256")
        != task.config["scheduler"]["cost_model"]["expected_sha256"]
    ):
        raise ValueError("frozen scheduler model hash differs from configuration")
    row_provenance = final["scheduler_model"].get("row_provenance", {})
    expected_row_provenance = {
        "total_accepted_rows": 6000,
        "model_fitting_rows": 4000,
        "untouched_evaluation_rows": 2000,
        "seed_27_overlap_with_fitting_or_selection_rows": 0,
        "seed_27_row_hash_in_coefficient_fitting_rows": False,
        "normalization_uses_seed_27": False,
        "coefficient_fitting_uses_seed_27": False,
        "regression_family_selection_uses_seed_27": False,
        "feature_selection_uses_seed_27": False,
        "hyperparameter_selection_uses_seed_27": False,
    }
    if any(row_provenance.get(name) != value for name, value in expected_row_provenance.items()):
        raise ValueError("frozen scheduler row provenance is incomplete or incompatible")
    if final.get("aggregation_tensor_policy") != aggregation_tensor_policy():
        raise ValueError("flat and hierarchical aggregation tensor policy evidence is incompatible")
    if final.get("configuration_id") != configuration_identity(task.config):
        raise ValueError("resolved configuration identity differs from the manifest")
    resource_allocation = measurements.get("resource_allocation")
    required_allocation = (
        "job_id",
        "array_job_id",
        "array_task_id",
        "partition",
        "allocated_gpus_on_node",
        "allocated_gpu_uuids",
        "allocated_nodes",
        "node_list",
        "cpus_per_task",
        "within_allocation_execution_order",
        "treatment_position",
    )
    if not isinstance(resource_allocation, dict) or any(
        not resource_allocation.get(name) for name in required_allocation
    ):
        raise ValueError("Slurm resource-allocation provenance is incomplete")
    expected_partition = "gpumedium"
    expected_nodes = "1" if collection == "scheduling_evaluation" else "2"
    expected_gpus_per_node = "4" if collection == "scheduling_evaluation" else "2"
    expected_cpus = "288" if collection == "scheduling_evaluation" else "144"
    if (
        resource_allocation["partition"] != expected_partition
        or resource_allocation["allocated_nodes"] != expected_nodes
        or resource_allocation["allocated_gpus_on_node"] != expected_gpus_per_node
        or resource_allocation["cpus_per_task"] != expected_cpus
    ):
        raise ValueError("Slurm resource allocation differs from the declared matrix")
    treatment = (
        task.config["scheduler"]["strategy"]
        if collection == "scheduling_evaluation"
        else task.config["aggregation_execution"]["topology"]
    )
    expected_order = _expected_treatment_order(collection, task.seed)
    if (
        tuple(resource_allocation["within_allocation_execution_order"].split(",")) != expected_order
        or int(resource_allocation["treatment_position"]) != expected_order.index(treatment) + 1
    ):
        raise ValueError("within-allocation treatment order provenance is incompatible")
    if environment.get("cuda_runtime") is None or environment.get("nccl") is None:
        raise ValueError("CUDA or NCCL version provenance is incomplete")
    process_mapping = measurements.get("process_mapping")
    process_count = task.config["parallel_execution"]["process_count"]
    try:
        allocation_uuids = allocated_gpu_uuids(resource_allocation["allocated_gpu_uuids"], expected_count=4)
    except RuntimeError as error:
        raise ValueError("allocation GPU UUID provenance is incomplete") from error
    devices_per_node = task.config["parallel_execution"]["devices_per_node"]
    try:
        process_uuids = [
            canonical_gpu_uuid(value.get("gpu_uuid"))
            for value in process_mapping
        ]
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("hardware process mapping or GPU UUID provenance is incomplete") from error
    if (
        not isinstance(process_mapping, list)
        or len(process_mapping) != process_count
        or {value.get("rank") for value in process_mapping} != set(range(process_count))
        or any(not value.get("host") or not value.get("gpu_uuid") for value in process_mapping)
        or len(set(process_uuids)) != 4
        or set(process_uuids) != set(allocation_uuids)
    ):
        raise ValueError("hardware process mapping or GPU UUID provenance is incomplete")
    for value in process_mapping:
        rank = value["rank"]
        expected_local_rank = rank % devices_per_node
        if (
            value.get("node_rank") != rank // devices_per_node
            or value.get("local_rank") != expected_local_rank
            or value.get("device_index") != expected_local_rank
        ):
            raise ValueError("process rank, node, device, and GPU UUID mapping is incompatible")
    for record in rounds:
        assignments = record.get("client_assignments")
        selected = record.get("selected_client_ids")
        if not isinstance(assignments, list) or [value.get("client_id") for value in assignments] != selected:
            raise ValueError("round assignments do not preserve selected-client order")
        if [value.get("selected_position") for value in assignments] != list(range(len(selected))):
            raise ValueError("round assignments have missing or duplicate selected positions")
        scheduler = record.get("scheduler")
        strategy = task.config["scheduler"]["strategy"]
        if (
            not isinstance(scheduler, dict)
            or scheduler.get("strategy") != strategy
            or scheduler.get("total_scheduler_seconds", -1) < 0
            or scheduler.get("model_sha256") != task.config["scheduler"]["cost_model"]["expected_sha256"]
            or not scheduler.get("model_provenance_identity")
        ):
            raise ValueError("scheduler timing is missing or invalid")
        expected_availability = {
            name: strategy == "event_structure_longest_processing_time" for name in EVENT_STRUCTURE_FEATURES
        }
        if scheduler.get("feature_availability") != expected_availability:
            raise ValueError("scheduler feature-availability evidence is incompatible")
        privacy = scheduler.get("privacy_metadata")
        if not isinstance(privacy, list) or any(
            value.get("field") not in EVENT_STRUCTURE_FEATURES
            or value.get("contains_label_information") is not False
            or value.get("raw_events_leave_client") is not False
            for value in privacy
        ):
            raise ValueError("scheduler privacy metadata contains an unpermitted field")
        expected_privacy_count = {
            "round_robin": 0,
            "example_count_longest_processing_time": len(selected),
            "event_structure_longest_processing_time": (len(selected) * len(EVENT_STRUCTURE_FEATURES)),
        }[strategy]
        if len(privacy) != expected_privacy_count or (
            privacy and {value.get("client_id") for value in privacy} != set(selected)
        ):
            raise ValueError("scheduler privacy metadata coverage is incomplete")
        if strategy == "event_structure_longest_processing_time" and any(
            {value["field"] for value in privacy if value.get("client_id") == client_id}
            != set(EVENT_STRUCTURE_FEATURES)
            for client_id in selected
        ):
            raise ValueError("event-structure privacy metadata fields are incomplete")
        expected_cost_source = {
            "round_robin": "selected_position",
            "example_count_longest_processing_time": "training_example_count",
            "event_structure_longest_processing_time": ("frozen_event_structure_wall_time_prediction"),
        }[strategy]
        if any(
            value.get("cost_source") != expected_cost_source
            or _nonnegative(value.get("cost"), "scheduler cost") < 0
            or (
                strategy == "event_structure_longest_processing_time"
                and set(value.get("features", {})) != set(EVENT_STRUCTURE_FEATURES)
            )
            for value in assignments
        ):
            raise ValueError("scheduler assignment cost evidence is incompatible")
        movement_fields = (
            "logical_intra_node_bytes",
            "logical_inter_node_bytes",
            "predicted_logical_inter_node_bytes",
            "client_result_collection_bytes",
            "model_distribution_bytes",
            "model_sized_payloads_crossing_node_boundaries",
        )
        if any(
            not isinstance(record.get(name), int) or isinstance(record.get(name), bool) or record[name] < 0
            for name in movement_fields
        ):
            raise ValueError("logical movement accounting is missing or invalid")
        if (
            record["logical_inter_node_bytes"] != record["predicted_logical_inter_node_bytes"]
            or record["client_result_collection_bytes"]
            != record["logical_intra_node_bytes"] + record["logical_inter_node_bytes"]
        ):
            raise ValueError("logical movement accounting differs from its declared prediction")
        expected_payload_count = (
            task.config["parallel_execution"]["node_count"] - 1
            if task.config["aggregation_execution"]["topology"] == "node_hierarchical"
            else sum(value["assigned_node_rank"] != 0 for value in assignments)
        )
        if record["model_sized_payloads_crossing_node_boundaries"] != expected_payload_count:
            raise ValueError("model-sized inter-node payload count is incompatible")
    test = final["test"]
    internal_duration = _positive(
        final.get("internal_treatment_duration_seconds"),
        "internal treatment duration",
    )
    derived_exposure = _positive(
        final.get("derived_treatment_gpu_exposure_hours"),
        "derived treatment GPU exposure",
    )
    if not math.isclose(
        derived_exposure,
        internal_duration * task.config["parallel_execution"]["device_count"] / 3600.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("derived treatment GPU exposure differs from duration and GPU count")
    execution_start = _finite(measurements.get("execution_start_unix_seconds"), "execution start")
    execution_end = _finite(measurements.get("execution_end_unix_seconds"), "execution end")
    if execution_end <= execution_start or not math.isclose(
        execution_end - execution_start,
        internal_duration,
        rel_tol=0.0,
        abs_tol=ALLOCATION_RECONCILIATION_TOLERANCE_SECONDS,
    ):
        raise ValueError("internal treatment duration does not match execution boundaries")
    return {
        "dataset": task.dataset,
        "seed": task.seed,
        "experiment": task.experiment,
        "strategy": task.config["scheduler"]["strategy"],
        "aggregation_topology": task.config["aggregation_execution"]["topology"],
        "run_directory": str(path),
        "official_test_accuracy": _fraction(test["accuracy"], "official-test accuracy"),
        "official_test_macro_f1": _fraction(test["macro_f1"], "official-test macro-F1"),
        "selected_round": int(final["selected_round"]),
        "total_runtime_seconds": sum(_positive(value["total_round_time_seconds"], "round time") for value in rounds),
        "mean_round_time_seconds": statistics.mean(
            _positive(value["total_round_time_seconds"], "round time") for value in rounds
        ),
        "client_wall_time_seconds": sum(
            _nonnegative(value["parallel_client_training_wall_time_seconds"], "client wall time") for value in rounds
        ),
        "scheduler_overhead_seconds": sum(
            _nonnegative(value["scheduler"]["total_scheduler_seconds"], "scheduler overhead") for value in rounds
        ),
        "scheduler_overhead_fraction": _fraction(final["scheduler_overhead_fraction"], "scheduler overhead fraction"),
        "predicted_load_imbalance": statistics.mean(
            _fraction(value["predicted_load_imbalance"], "predicted load imbalance") for value in rounds
        ),
        "observed_load_imbalance": statistics.mean(
            _fraction(value["process_load_imbalance"], "observed load imbalance") for value in rounds
        ),
        "aggregation_time_seconds": sum(
            _nonnegative(value["aggregation_time_seconds"], "aggregation time") for value in rounds
        ),
        "node_local_reduction_time_seconds": sum(
            _nonnegative(value.get("node_local_reduction_time_seconds", 0.0), "node-local reduction time")
            for value in rounds
        ),
        "inter_node_movement_time_seconds": sum(
            _nonnegative(value.get("inter_node_contribution_movement_time_seconds", 0.0), "inter-node time")
            for value in rounds
        ),
        "global_reduction_time_seconds": sum(
            _nonnegative(value.get("global_reduction_time_seconds", 0.0), "global reduction time") for value in rounds
        ),
        "logical_intra_node_bytes": sum(int(value.get("logical_intra_node_bytes", 0)) for value in rounds),
        "logical_inter_node_bytes": sum(int(value.get("logical_inter_node_bytes", 0)) for value in rounds),
        "predicted_logical_inter_node_bytes": sum(
            int(value.get("predicted_logical_inter_node_bytes", 0)) for value in rounds
        ),
        "client_result_collection_bytes": sum(int(value.get("client_result_collection_bytes", 0)) for value in rounds),
        "model_distribution_bytes": sum(int(value.get("model_distribution_bytes", 0)) for value in rounds),
        "model_sized_payloads_crossing_node_boundaries": sum(
            int(value.get("model_sized_payloads_crossing_node_boundaries", 0)) for value in rounds
        ),
        "maximum_peak_allocated_bytes": max(
            value
            for record in measurements["rounds"]
            for value in record.get("peak_cuda_memory_bytes_by_process_rank", {}).values()
        ),
        "maximum_peak_reserved_bytes": max(
            value
            for record in measurements["rounds"]
            for value in record.get("peak_cuda_reserved_bytes_by_process_rank", {}).values()
        ),
        "gpu_utilization_percent": 100
        * _fraction(
            _finite(measurements.get("gpu_utilization", {}).get("mean_percent"), "GPU utilization") / 100,
            "GPU utilization fraction",
        ),
        "internal_treatment_duration_seconds": internal_duration,
        "derived_treatment_gpu_exposure_hours": derived_exposure,
        "execution_start_unix_seconds": execution_start,
        "execution_end_unix_seconds": execution_end,
        "selected_clients": [value["selected_client_ids"] for value in rounds],
        "client_seeds": [
            (value["round_number"], value["client_id"], value["resolved_training_seed"]) for value in clients
        ],
        "client_example_counts": [value["client_example_counts"] for value in rounds],
        "client_training_examples_presented": [value["client_training_examples_presented"] for value in rounds],
        "update_identities": [value["ordered_update_identities"] for value in rounds],
        "aggregate_model_identities": [value["global_model_identity_after_aggregation"] for value in rounds],
        "aggregation_weights": [value["aggregation_weights"] for value in rounds],
        "aggregation_orders": [value["aggregation_order"] for value in rounds],
        "validation_records": [
            (
                value["validation_loss"],
                value["validation_accuracy"],
                value["validation_macro_f1"],
                value["selected_checkpoint"],
            )
            for value in rounds
        ],
        "official_predictions": test.get("predictions"),
        "official_test_access_count": final["data_protocol"]["official_test_access_count"],
        "scientific_identity": final["scientific_identity"],
        "split_id": final["split_id"],
        "partition_id": final["partition_id"],
        "model_initialization_id": final["model_initialization_id"],
        "logical_communication": final["logical_communication"],
        "git_commit": acceptance.get("execution_provenance", {}).get("git_commit"),
        "git_record": git_record,
        "environment": environment,
        "resource_allocation": resource_allocation,
        "process_mapping": process_mapping,
        "scheduler_model": final["scheduler_model"],
        "permitted_pre_execution_information_only": True,
        "round_measurements": [
            {
                "round_number": value["round_number"],
                "total_round_time_seconds": value["total_round_time_seconds"],
                "aggregation_time_seconds": value["aggregation_time_seconds"],
                "process_load_imbalance": value["process_load_imbalance"],
                "predicted_load_imbalance": value["predicted_load_imbalance"],
                "process_busy_time_seconds": value["process_busy_time_seconds"],
                "estimated_idle_time_seconds_by_process_rank": value["estimated_idle_time_seconds_by_process_rank"],
                "scheduler_overhead_fraction": value.get("scheduler_overhead_fraction"),
                "logical_inter_node_bytes": value.get("logical_inter_node_bytes", 0),
                "client_assignments": value["client_assignments"],
            }
            for value in rounds
        ],
        "_state": _state(path, final),
    }


def _stats(values: list[float]) -> dict:
    if len(values) != 3:
        return {}
    return {
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _reconcile_allocation(record: dict, runs: list[dict], collection: str) -> dict:
    expected_order = _expected_treatment_order(collection, runs[0]["seed"])
    treatment_name = "strategy" if collection == "scheduling_evaluation" else "aggregation_topology"
    ordered = sorted(runs, key=lambda value: int(value["resource_allocation"]["treatment_position"]))
    if len(ordered) != len(expected_order) or tuple(value[treatment_name] for value in ordered) != expected_order:
        raise ValueError(f"Slurm allocation {record['job_id']} does not contain its exact treatment order")
    if len({(value["dataset"], value["seed"]) for value in ordered}) != 1:
        raise ValueError(f"Slurm allocation {record['job_id']} crosses dataset or seed identities")

    allocation_start = record["start_unix_seconds"]
    allocation_end = record["end_unix_seconds"]
    initialization = ordered[0]["execution_start_unix_seconds"] - allocation_start
    between_intervals = []
    for previous, following in zip(ordered, ordered[1:], strict=False):
        between_intervals.append(
            following["execution_start_unix_seconds"] - previous["execution_end_unix_seconds"]
        )
    remaining = allocation_end - ordered[-1]["execution_end_unix_seconds"]
    values = [initialization, *between_intervals, remaining]
    if any(value < -ALLOCATION_RECONCILIATION_TOLERANCE_SECONDS for value in values):
        raise ValueError(f"Slurm allocation {record['job_id']} contains overlapping or out-of-bound treatments")
    initialization = max(0.0, initialization)
    between_intervals = [max(0.0, value) for value in between_intervals]
    remaining = max(0.0, remaining)
    treatment_duration = sum(value["internal_treatment_duration_seconds"] for value in ordered)
    component_total = treatment_duration + initialization + sum(between_intervals) + remaining
    reconciliation_error = record["slurm_allocation_elapsed_seconds"] - component_total
    if abs(reconciliation_error) > ALLOCATION_RECONCILIATION_TOLERANCE_SECONDS:
        raise ValueError(
            f"Slurm allocation {record['job_id']} reconciliation error exceeds "
            f"{ALLOCATION_RECONCILIATION_TOLERANCE_SECONDS} seconds"
        )
    for run in ordered:
        run["allocation_reconciliation_id"] = record["job_id"]
    return {
        "job_id": record["job_id"],
        "dataset": ordered[0]["dataset"],
        "seed": ordered[0]["seed"],
        "treatment_order": list(expected_order),
        "slurm_allocation_elapsed_seconds": record["slurm_allocation_elapsed_seconds"],
        "slurm_allocation_gpu_hours": record["slurm_allocation_gpu_hours"],
        "allocated_gpu_count": record["allocated_gpu_count"],
        "allocation_initialization_seconds": initialization,
        "between_treatment_overhead_seconds": sum(between_intervals),
        "between_treatment_intervals_seconds": between_intervals,
        "remaining_allocation_overhead_seconds": remaining,
        "allocation_reconciliation_error_seconds": reconciliation_error,
        "allocation_reconciliation_tolerance_seconds": ALLOCATION_RECONCILIATION_TOLERANCE_SECONDS,
        "treatments": [
            {
                "treatment": value[treatment_name],
                "treatment_position": int(value["resource_allocation"]["treatment_position"]),
                "internal_treatment_duration_seconds": value["internal_treatment_duration_seconds"],
                "derived_treatment_gpu_exposure_hours": value["derived_treatment_gpu_exposure_hours"],
            }
            for value in ordered
        ],
        "billing_scope": "one Slurm allocation; not duplicated across sequential treatments",
        "derived_treatment_gpu_exposure_billed_separately": False,
        "state": record["state"],
        "exit_code": record["exit_code"],
        "allocation": record["allocation"],
        "start": record["start"],
        "end": record["end"],
    }


def _structural(reference: dict, candidate: dict, *, hierarchical: bool) -> tuple[bool, list[str]]:
    fields = [
        "scientific_identity",
        "split_id",
        "partition_id",
        "model_initialization_id",
        "selected_clients",
        "client_seeds",
        "client_example_counts",
        "client_training_examples_presented",
        "update_identities",
        "aggregation_weights",
        "aggregation_orders",
        "logical_communication",
    ]
    if not hierarchical:
        fields.append("aggregate_model_identities")
    differences = [field for field in fields if reference[field] != candidate[field]]
    return not differences, differences


def _paired(reference: dict, candidate: dict, *, hierarchical: bool, config: dict) -> dict:
    structural, differences = _structural(reference, candidate, hierarchical=hierarchical)
    numerical = classify_model_states(
        reference["_state"],
        candidate["_state"],
        absolute_tolerance=float(config["aggregation_execution"]["absolute_tolerance"]),
        relative_tolerance=float(config["aggregation_execution"]["relative_tolerance"]),
    )
    predictions_equal = prediction_identity(reference["official_predictions"], candidate["official_predictions"])
    checkpoint_equal = reference["selected_round"] == candidate["selected_round"]
    speedup = reference["total_runtime_seconds"] / candidate["total_runtime_seconds"]
    reduction = (reference["total_runtime_seconds"] - candidate["total_runtime_seconds"]) / reference[
        "total_runtime_seconds"
    ]
    scheduler_exact = (
        structural
        and numerical.bitwise_parameter_identity
        and (
            reference["validation_records"] == candidate["validation_records"]
            and checkpoint_equal
            and predictions_equal is True
            and reference["official_test_accuracy"] == candidate["official_test_accuracy"]
            and reference["official_test_macro_f1"] == candidate["official_test_macro_f1"]
        )
    )
    return {
        "dataset": candidate["dataset"],
        "seed": candidate["seed"],
        "reference": reference["strategy"] if not hierarchical else reference["aggregation_topology"],
        "treatment": candidate["strategy"] if not hierarchical else candidate["aggregation_topology"],
        "structural_identity": structural,
        "structural_differences": differences,
        **numerical.record(),
        "checkpoint_selection_identity": checkpoint_equal,
        "prediction_identity": predictions_equal,
        "official_test_accuracy_difference": candidate["official_test_accuracy"] - reference["official_test_accuracy"],
        "official_test_macro_f1_difference": candidate["official_test_macro_f1"] - reference["official_test_macro_f1"],
        "validation_record_identity": reference["validation_records"] == candidate["validation_records"],
        "scheduler_scientific_identity": scheduler_exact if not hierarchical else None,
        "paired_speedup": speedup,
        "paired_runtime_reduction": reduction,
        "logical_inter_node_reduction_bytes": reference["logical_inter_node_bytes"]
        - candidate["logical_inter_node_bytes"],
        "predicted_logical_inter_node_reduction_bytes": reference["predicted_logical_inter_node_bytes"]
        - candidate["predicted_logical_inter_node_bytes"],
        "runtime_regression_fraction": -reduction,
    }


def _decision(collection: str, runs: list[dict], pairs: list[dict], valid: bool) -> dict:
    if collection == "scheduling_evaluation":
        event_pairs = [value for value in pairs if value["treatment"] == "event_structure_longest_processing_time"]
        example_pairs = [value for value in pairs if value["treatment"] == "example_count_longest_processing_time"]
        required_pair_keys = {(dataset, seed) for dataset in ("shd", "ssc") for seed in EVALUATION_SEEDS}
        event_by_key = {(value["dataset"], value["seed"]): value for value in event_pairs}
        example_by_key = {(value["dataset"], value["seed"]): value for value in example_pairs}
        exact_event_coverage = len(event_pairs) == 6 and set(event_by_key) == required_pair_keys
        exact_example_coverage = len(example_pairs) == 6 and set(example_by_key) == required_pair_keys
        dataset_means = {
            dataset: (
                statistics.mean(event_by_key[(dataset, seed)]["paired_runtime_reduction"] for seed in EVALUATION_SEEDS)
                if exact_event_coverage
                else None
            )
            for dataset in ("shd", "ssc")
        }
        example_dataset_means = {
            dataset: (
                statistics.mean(
                    example_by_key[(dataset, seed)]["paired_runtime_reduction"] for seed in EVALUATION_SEEDS
                )
                if exact_example_coverage
                else None
            )
            for dataset in ("shd", "ssc")
        }
        improved_seed_counts = {
            dataset: (
                sum(
                    event_by_key[(dataset, seed)]["paired_runtime_reduction"] > 0 for seed in EVALUATION_SEEDS
                )
                if exact_event_coverage
                else None
            )
            for dataset in ("shd", "ssc")
        }
        conditions = {
            "exact_dataset_seed_pair_coverage": exact_event_coverage,
            "all_structural_and_scientific_equivalence": exact_event_coverage
            and all(value["scheduler_scientific_identity"] for value in event_pairs),
            "scheduler_overhead_below_one_percent": len(runs) == 18
            and all(value["scheduler_overhead_fraction"] < 0.01 for value in runs),
            "shd_runtime_improvement_at_least_five_percent": exact_event_coverage
            and dataset_means["shd"] >= 0.05,
            "ssc_runtime_improvement_at_least_five_percent": exact_event_coverage
            and dataset_means["ssc"] >= 0.05,
            "not_slower_than_example_count_each_dataset": exact_event_coverage
            and exact_example_coverage
            and all(dataset_means[dataset] >= example_dataset_means[dataset] for dataset in ("shd", "ssc")),
            "two_of_three_seeds_improve_each_dataset": exact_event_coverage
            and all(improved_seed_counts[dataset] >= 2 for dataset in ("shd", "ssc")),
            "no_dataset_seed_pair_more_than_two_percent_slower": exact_event_coverage
            and all(event_by_key[key]["paired_runtime_reduction"] >= -0.02 for key in required_pair_keys),
            "predictions_and_checkpoints_identical": exact_event_coverage
            and all(
                value["prediction_identity"] is True and value["checkpoint_selection_identity"] for value in event_pairs
            ),
            "permitted_pre_execution_information_only": len(runs) == 18
            and all(value["permitted_pre_execution_information_only"] for value in runs),
        }
        adopted = valid and all(conditions.values())
        return {
            "decision": "event_structure_scheduler_adopted" if adopted else "event_structure_scheduler_not_adopted",
            "evidence_available": valid,
            "pairing_semantics": {
                "pairing_unit": "dataset_seed",
                "datasets_pooled": False,
                "reference_treatment": "round_robin",
                "candidate_treatment": "event_structure_longest_processing_time",
                "required_pair_keys": [
                    {"dataset": dataset, "seed": seed}
                    for dataset in ("shd", "ssc")
                    for seed in EVALUATION_SEEDS
                ],
                "dataset_mean_definition": "arithmetic mean of the three paired runtime reductions within each dataset",
                "seed_improvement_definition": "paired runtime reduction is strictly greater than zero",
                "required_improved_seeds_per_dataset": 2,
                "minimum_allowed_paired_runtime_reduction": -0.02,
            },
            "paired_dataset_seed_runtime_reductions": [
                {
                    "dataset": dataset,
                    "seed": seed,
                    "paired_runtime_reduction": event_by_key[(dataset, seed)]["paired_runtime_reduction"],
                }
                for dataset in ("shd", "ssc")
                for seed in EVALUATION_SEEDS
                if (dataset, seed) in event_by_key
            ],
            "dataset_mean_paired_runtime_reductions": dataset_means,
            "example_count_dataset_mean_paired_runtime_reductions": example_dataset_means,
            "improved_seed_counts_by_dataset": improved_seed_counts,
            "conditions": conditions,
        }
    hierarchical_pairs = [value for value in pairs if value["treatment"] == "node_hierarchical"]
    hierarchical_runs = [value for value in runs if value["aggregation_topology"] == "node_hierarchical"]
    conditions = {
        "every_update_once": len(hierarchical_pairs) == 6
        and all(value["structural_identity"] for value in hierarchical_pairs),
        "weights_and_denominators_correct": len(hierarchical_pairs) == 6
        and all("aggregation_weights" not in value["structural_differences"] for value in hierarchical_pairs),
        "structural_and_mathematical_equivalence": len(hierarchical_pairs) == 6
        and all(value["structural_identity"] and value["mathematical_equivalence"] for value in hierarchical_pairs),
        "parameter_differences_within_tolerance": len(hierarchical_pairs) == 6
        and all(value["mathematical_equivalence"] for value in hierarchical_pairs),
        "official_predictions_agree": len(hierarchical_pairs) == 6
        and all(value["prediction_identity"] is True for value in hierarchical_pairs),
        "selected_checkpoints_agree": len(hierarchical_pairs) == 6
        and all(value["checkpoint_selection_identity"] for value in hierarchical_pairs),
        "logical_inter_node_movement_reduced": len(hierarchical_pairs) == 6
        and all(
            value["logical_inter_node_reduction_bytes"] > 0
            and value["logical_inter_node_reduction_bytes"] == value["predicted_logical_inter_node_reduction_bytes"]
            for value in hierarchical_pairs
        ),
        "no_material_runtime_regression": len(hierarchical_pairs) == 6
        and all(value["runtime_regression_fraction"] <= 0.02 for value in hierarchical_pairs),
        "official_test_ownership_preserved": len(hierarchical_runs) == 6
        and all(value["official_test_access_count"] == 1 for value in hierarchical_runs),
    }
    retained = valid and all(conditions.values())
    return {
        "decision": "node_hierarchical_reduction_retained" if retained else "node_hierarchical_reduction_not_retained",
        "evidence_available": valid,
        "conditions": conditions,
    }


def summarize_evaluation(
    manifest: str | Path,
    runs_root: str | Path,
    output_dir: str | Path,
    *,
    slurm_accounting: str | Path | None = None,
    data_root: str | Path | None = None,
) -> dict:
    tasks = load_evaluation_manifest(manifest)
    tasks = _materialize_runtime_data_root(tasks, data_root)
    collection = tasks[0].config["evaluation"]["collection"]
    root = Path(runs_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    findings = []
    runs = []
    for task in tasks:
        try:
            runs.append(_load_run(task, root, collection))
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            findings.append(f"{task.experiment} seed {task.seed}: {error}")
    accounting = {}
    accounting_hash = None
    accounting_identifier_field = None
    if slurm_accounting is None:
        findings.append("Slurm accounting was not supplied")
    else:
        try:
            accounting, accounting_hash, accounting_identifier_field = _slurm_accounting(slurm_accounting)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            findings.append(f"Slurm accounting: {error}")
    used_accounting = set()
    for run in runs:
        allocation = run["resource_allocation"]
        job_id = f"{allocation['array_job_id']}_{allocation['array_task_id']}"
        record = accounting.get(job_id)
        if record is None:
            findings.append(f"{run['experiment']} seed {run['seed']}: Slurm accounting record is missing")
            continue
        used_accounting.add(job_id)
        if record["state"] != "COMPLETED" or record["exit_code"] != "0:0":
            findings.append(f"Slurm allocation {job_id} did not complete successfully")
        if record["allocated_gpu_count"] != 4:
            findings.append(f"Slurm allocation {job_id} did not allocate exactly four GPUs")
        run["slurm_allocation_id"] = job_id
    for job_id in sorted(set(accounting).difference(used_accounting)):
        findings.append(f"Slurm accounting contains unexpected array task {job_id}")
    reconciled_allocations = []
    for job_id in sorted(used_accounting):
        allocation_runs = [value for value in runs if value.get("slurm_allocation_id") == job_id]
        try:
            reconciled_allocations.append(_reconcile_allocation(accounting[job_id], allocation_runs, collection))
        except (KeyError, TypeError, ValueError) as error:
            findings.append(str(error))
    by_key = {
        (
            value["dataset"],
            value["seed"],
            value["strategy"] if collection == "scheduling_evaluation" else value["aggregation_topology"],
        ): value
        for value in runs
    }
    reference_name = "round_robin" if collection == "scheduling_evaluation" else "flat_ordered"
    treatments = (
        ("example_count_longest_processing_time", "event_structure_longest_processing_time")
        if collection == "scheduling_evaluation"
        else ("node_hierarchical",)
    )
    pairs = []
    for dataset in ("shd", "ssc"):
        for seed in EVALUATION_SEEDS:
            reference = by_key.get((dataset, seed, reference_name))
            for treatment in treatments:
                candidate = by_key.get((dataset, seed, treatment))
                if reference is None or candidate is None:
                    continue
                reference_configuration = candidate_config(tasks, dataset, seed, reference_name, collection)
                treatment_configuration = candidate_config(tasks, dataset, seed, treatment, collection)
                pair = _paired(
                    reference,
                    candidate,
                    hierarchical=collection == "hierarchical_reduction_evaluation",
                    config=treatment_configuration,
                )
                pair["resolved_configuration_comparison"] = validate_resolved_evaluation_pair(
                    reference_configuration,
                    treatment_configuration,
                    collection,
                )
                pairs.append(pair)
    expected = 18 if collection == "scheduling_evaluation" else 12
    expected_pairs = 12 if collection == "scheduling_evaluation" else 6
    group_name = "strategy" if collection == "scheduling_evaluation" else "aggregation_topology"
    groups = []
    for dataset in ("shd", "ssc"):
        values = sorted({value[group_name] for value in runs if value["dataset"] == dataset})
        for treatment in values:
            selected = [value for value in runs if value["dataset"] == dataset and value[group_name] == treatment]
            metric_names = (
                "official_test_accuracy",
                "official_test_macro_f1",
                "total_runtime_seconds",
                "mean_round_time_seconds",
                "client_wall_time_seconds",
                "scheduler_overhead_seconds",
                "scheduler_overhead_fraction",
                "predicted_load_imbalance",
                "observed_load_imbalance",
                "aggregation_time_seconds",
                "node_local_reduction_time_seconds",
                "inter_node_movement_time_seconds",
                "global_reduction_time_seconds",
                "logical_intra_node_bytes",
                "logical_inter_node_bytes",
                "predicted_logical_inter_node_bytes",
                "client_result_collection_bytes",
                "model_distribution_bytes",
                "model_sized_payloads_crossing_node_boundaries",
                "maximum_peak_allocated_bytes",
                "maximum_peak_reserved_bytes",
                "gpu_utilization_percent",
                "internal_treatment_duration_seconds",
                "derived_treatment_gpu_exposure_hours",
            )
            metrics = {
                name: _stats([float(value[name]) for value in selected])
                for name in metric_names
                if all(name in value for value in selected)
            }
            selected_pairs = [
                value for value in pairs if value["dataset"] == dataset and value["treatment"] == treatment
            ]
            for name in (
                "paired_speedup",
                "paired_runtime_reduction",
                "maximum_absolute_parameter_difference",
                "maximum_relative_parameter_difference",
                "runtime_regression_fraction",
            ):
                if selected_pairs:
                    metrics[name] = _stats([float(value[name]) for value in selected_pairs])
            groups.append(
                {
                    "dataset": dataset,
                    group_name: treatment,
                    "seeds_completed": sorted(value["seed"] for value in selected),
                    "metrics": metrics,
                    "paired_numerical_classifications": [
                        {
                            key: value[key]
                            for key in (
                                "seed",
                                "structural_identity",
                                "mathematical_equivalence",
                                "bitwise_parameter_identity",
                                "prediction_identity",
                                "checkpoint_selection_identity",
                            )
                        }
                        for value in selected_pairs
                    ],
                }
            )
    public_runs = [{key: value for key, value in run.items() if not key.startswith("_")} for run in runs]
    scheduler_equivalence = (
        collection != "scheduling_evaluation"
        or len(pairs) == expected_pairs
        and all(value["scheduler_scientific_identity"] for value in pairs)
    )
    aggregation_equivalence = (
        collection != "hierarchical_reduction_evaluation"
        or len(pairs) == expected_pairs
        and all(value["structural_identity"] and value["mathematical_equivalence"] for value in pairs)
    )
    provenance_completeness = (
        len(runs) == expected
        and len(used_accounting) == 6
        and all(
            value["git_commit"]
            and value["git_record"].get("commit") == value["git_commit"]
            and value["scheduler_model"].get("model_sha256")
            and value.get("allocation_reconciliation_id")
            and value["environment"].get("cuda_runtime")
            and value["environment"].get("nccl")
            for value in runs
        )
    )
    collection_valid = (
        len(runs) == expected
        and not findings
        and scheduler_equivalence
        and aggregation_equivalence
        and provenance_completeness
        and all(value["official_test_access_count"] == 1 for value in runs)
    )
    decision = _decision(collection, runs, pairs, collection_valid)
    acceptance = {
        "execution_completion": len(runs) == expected,
        "measurement_completeness": not findings,
        "scheduler_equivalence": scheduler_equivalence,
        "aggregation_equivalence": aggregation_equivalence,
        "official_test_isolation": len(runs) == expected
        and all(value["official_test_access_count"] == 1 for value in runs),
        "provenance_completeness": provenance_completeness,
        "hypothesis_decision": decision["decision"],
        "collection_valid": collection_valid,
        "validation_findings": findings,
    }
    stem = collection
    unique_allocations = reconciled_allocations
    summary = {
        "schema_version": 1,
        "collection": collection,
        "valid": collection_valid,
        "expected_task_count": expected,
        "completed_task_count": len(runs),
        "required_seeds": list(EVALUATION_SEEDS),
        "datasets_pooled": False,
        "groups": groups,
        "paired_records": pairs,
        "allocations": unique_allocations,
        "allocation_count": len(unique_allocations),
        "total_slurm_allocation_gpu_hours": sum(
            value["slurm_allocation_gpu_hours"] for value in unique_allocations
        ),
        "runs": public_runs,
        "acceptance": acceptance,
        "decision": decision,
        "validation_findings": findings,
    }
    (output / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / f"{stem}_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / f"{stem}_acceptance.json").write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / f"{stem}_validation_findings.json").write_text(
        json.dumps({"findings": findings}, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "collection": collection,
        "run_count": len(runs),
        "allocation_count": len(unique_allocations),
        "git_commits": sorted({value["git_commit"] for value in runs if value["git_commit"]}),
        "model_hashes": sorted({value["scheduler_model"]["model_sha256"] for value in runs}),
        "runtime_dataset_roots": {
            dataset: next(task.config["dataset"]["root"] for task in tasks if task.dataset == dataset)
            for dataset in ("shd", "ssc")
        },
        "slurm_accounting_job_id_field": accounting_identifier_field,
        "slurm_accounting_sha256": accounting_hash,
    }
    (output / f"{stem}_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output / f"{stem}_summary.csv", groups)
    _write_markdown(output / f"{stem}_summary.md", summary)
    return summary


def candidate_config(tasks, dataset: str, seed: int, treatment: str, collection: str) -> dict:
    key = "strategy" if collection == "scheduling_evaluation" else "topology"
    section = "scheduler" if collection == "scheduling_evaluation" else "aggregation_execution"
    return next(
        task.config
        for task in tasks
        if task.dataset == dataset and task.seed == seed and task.config[section][key] == treatment
    )


def _write_csv(path: Path, groups: list[dict]) -> None:
    rows = []
    for group in groups:
        row = {key: value for key, value in group.items() if key != "metrics"}
        row["seeds_completed"] = ",".join(str(value) for value in group["seeds_completed"])
        for metric, values in group["metrics"].items():
            for statistic, value in values.items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: dict) -> None:
    lines = [
        f"# {summary['collection'].replace('_', ' ').title()}",
        "",
        f"Status: **{'valid' if summary['valid'] else 'invalid'}**",
        "",
        "Datasets are summarized separately; three seeds do not establish statistical significance.",
        "",
        f"Decision: `{summary['decision']['decision']}`",
        "",
        "## Grouped measurements",
        "",
    ]
    group_name = "strategy" if summary["collection"] == "scheduling_evaluation" else "aggregation_topology"
    for group in summary["groups"]:
        lines.extend(
            [
                f"### {group['dataset'].upper()} — {group[group_name]}",
                "",
                f"Seeds completed: {', '.join(str(value) for value in group['seeds_completed'])}",
                "",
            ]
        )
        for metric, values in group["metrics"].items():
            if values:
                lines.append(
                    f"- {metric}: mean {values['mean']:.12g}; sample SD "
                    f"{values['sample_standard_deviation']:.12g}; range "
                    f"[{values['minimum']:.12g}, {values['maximum']:.12g}]"
                )
        lines.append("")
    lines.extend(
        [
            "## Paired comparisons",
            "",
            "| Dataset | Seed | Reference | Treatment | Speedup | Runtime reduction | "
            "Structural | Mathematical | Bitwise | Maximum absolute difference | "
            "Prediction identity | Checkpoint identity |",
            "|---|---:|---|---|---:|---:|---|---|---|---:|---|---|",
        ]
    )
    for record in summary["paired_records"]:
        lines.append(
            f"| {record['dataset'].upper()} | {record['seed']} | {record['reference']} | "
            f"{record['treatment']} | {record['paired_speedup']:.12g} | "
            f"{record['paired_runtime_reduction']:.12g} | {record['structural_identity']} | "
            f"{record['mathematical_equivalence']} | {record['bitwise_parameter_identity']} | "
            f"{record['maximum_absolute_parameter_difference']:.12g} | "
            f"{record['prediction_identity']} | {record['checkpoint_selection_identity']} |"
        )
    lines.extend(
        [
            "",
            "## Decision conditions",
            "",
        ]
    )
    lines.extend(f"- {name}: {value}" for name, value in summary["decision"]["conditions"].items())
    if summary["validation_findings"]:
        lines.extend(["", "## Validation findings", ""])
        lines.extend(f"- {value}" for value in summary["validation_findings"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
