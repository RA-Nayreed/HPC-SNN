"""Single-node synchronous FedAvg across exclusive or MPS-served CUDA processes."""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from fedapfa.configuration import distributed_execution_identity, distributed_scientific_identity
from fedapfa.distributed.assignment_broadcast import (
    assign_clients,
    broadcast_selected_clients,
)
from fedapfa.distributed.client_worker import order_client_results, train_rank_clients
from fedapfa.distributed.collectives import broadcast_model_state, gather_rank_payloads
from fedapfa.distributed.process_context import ProcessContext
from fedapfa.federated.acceptance import evaluate_federated_acceptance
from fedapfa.federated.aggregation import clone_state_dict
from fedapfa.federated.checkpointing import (
    configuration_identity,
    load_federated_checkpoint,
    read_git_commit,
    save_federated_checkpoint,
    state_identity,
)
from fedapfa.federated.client import evaluate_model, reset_snn_state, synchronize_cuda
from fedapfa.federated.client_sampling import ClientSelectionSchedule
from fedapfa.federated.communication_accounting import communication_for_clients, model_payload_bytes
from fedapfa.federated.fedavg import aggregate_client_results
from fedapfa.federated.randomness import derive_seed
from fedapfa.federated.round_state import RoundResult
from fedapfa.federated.server import global_model_norm, validate_global_model
from fedapfa.metrics.client_fairness import fairness_proxy_record
from fedapfa.training.federated import (
    _load_official_test_record,
    _logger,
    _mean_client_spike_rates,
    _verify_or_write,
    _write_jsonl,
)
from fedapfa.utilities.serialization import atomic_write_json


def _round_process_records(payloads):
    clients = {str(value.process_rank): list(value.assigned_client_ids) for value in payloads}
    busy = {str(value.process_rank): value.process_busy_time_seconds for value in payloads}
    memory = {str(value.process_rank): value.peak_cuda_memory_bytes for value in payloads}
    reserved = {str(value.process_rank): value.peak_cuda_reserved_bytes for value in payloads}
    examples = {str(value.process_rank): value.assigned_example_count for value in payloads}
    return clients, busy, memory, reserved, examples


def _assignment_records(assignments, process_records: list[dict]) -> list[dict]:
    mappings = {value["rank"]: value for value in process_records}
    return [
        {
            **assignment.record(),
            "device_index": mappings[assignment.process_rank]["device_index"],
            "device_slot": mappings[assignment.process_rank]["device_slot"],
        }
        for assignment in assignments
    ]


def _stable_process_mapping(process_records: list[dict]) -> list[dict]:
    excluded = {
        "host",
        "process_resident_memory_bytes",
        "process_resident_memory_before_workload_bytes",
        "process_resident_memory_after_workload_bytes",
        "workload_resident_memory_delta_bytes",
    }
    return [
        {key: value for key, value in process_record.items() if key not in excluded}
        for process_record in process_records
    ]


def _start_round_profiler(config: dict, context: ProcessContext, path: Path, round_number: int):
    measurement = config["execution_measurement"]
    if not measurement["profiler_enabled"] or round_number not in measurement["profiled_rounds"]:
        return None
    activities = [torch.profiler.ProfilerActivity.CPU]
    if context.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler = torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True)
    profiler.start()
    trace_dir = path / "profiles" / f"rank_{context.rank}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    return profiler, trace_dir / f"round_{round_number}.json"


def _stop_round_profiler(active) -> None:
    if active is None:
        return
    profiler, trace_path = active
    profiler.step()
    profiler.stop()
    profiler.export_chrome_trace(str(trace_path))


def _synchronize_processes(context: ProcessContext) -> None:
    with torch.profiler.record_function("process_round_synchronization"):
        synchronize_cuda(context.device)
        dist.barrier()


def _load_measurements(path: Path, identity: dict) -> dict:
    measurement_path = path / "execution_measurements.json"
    if not measurement_path.is_file():
        return {
            "schema_version": 1,
            "execution_identity": identity,
            "resume_count": 0,
            "completed": False,
            "rounds": [],
            "official_test_time_seconds": None,
        }
    value = json.loads(measurement_path.read_text(encoding="utf-8"))
    if value.get("execution_identity") != identity:
        raise RuntimeError("stored distributed execution identity is incompatible")
    value["resume_count"] = int(value.get("resume_count", 0)) + 1
    return value


def _gpu_utilization_record(telemetry_path: str | None) -> dict | None:
    if telemetry_path is None or not Path(telemetry_path).is_file():
        return None
    by_device: dict[str, list[float]] = {}
    with Path(telemetry_path).open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) < 5:
                continue
            try:
                device_index = str(int(row[1].strip()))
                utilization = float(row[4].strip())
            except ValueError:
                continue
            if not 0.0 <= utilization <= 100.0:
                continue
            by_device.setdefault(device_index, []).append(utilization)
    values = [value for device_values in by_device.values() for value in device_values]
    if not values:
        return None
    return {
        "source": "nvidia-smi job-level physical-device samples",
        "sample_count": len(values),
        "mean_percent": statistics.fmean(values),
        "minimum_percent": min(values),
        "maximum_percent": max(values),
        "by_device_index": {
            device: {
                "sample_count": len(device_values),
                "mean_percent": statistics.fmean(device_values),
                "minimum_percent": min(device_values),
                "maximum_percent": max(device_values),
            }
            for device, device_values in sorted(by_device.items(), key=lambda item: int(item[0]))
        },
    }


def _required_gpu_utilization_record(config: dict, telemetry_path: str | None) -> dict:
    utilization = _gpu_utilization_record(telemetry_path)
    if utilization is None:
        raise RuntimeError("configured physical-device utilization measurements are unavailable")
    expected_devices = {str(value) for value in range(config["parallel_execution"]["device_count"])}
    if set(utilization["by_device_index"]) != expected_devices:
        raise RuntimeError("physical-device utilization samples do not cover the configured devices")
    utilization["sampling_interval_seconds"] = config["execution_measurement"][
        "utilization_interval_seconds"
    ]
    return utilization


def _verify_distributed_records(config: dict, round_records: list[dict], measurements: dict) -> list[str]:
    failures: list[str] = []
    expected_world = config["parallel_execution"]["process_count"]
    selected_count = config["federated"]["clients_per_round"]
    for record in round_records:
        assignments = record.get("client_assignments")
        if not isinstance(assignments, list) or len(assignments) != selected_count:
            failures.append(f"round {record.get('round_number')} has invalid client assignments")
            continue
        expected = [position % expected_world for position in range(selected_count)]
        if [value.get("process_rank") for value in assignments] != expected:
            failures.append(f"round {record.get('round_number')} has incompatible round-robin assignment")
        selected = record.get("selected_client_ids", [])
        if [value.get("client_id") for value in assignments] != selected:
            failures.append(f"round {record.get('round_number')} assignment order differs from client selection")
        if record.get("ordered_update_identities") is None or len(record["ordered_update_identities"]) != len(
            selected
        ):
            failures.append(f"round {record.get('round_number')} lacks ordered update identities")
        if record.get("parallel_execution", {}).get("process_count") != expected_world:
            failures.append(f"round {record.get('round_number')} has incompatible process count")
    if len(measurements.get("rounds", [])) != len(round_records):
        failures.append("execution measurement count differs from communication-round count")
    return failures


def _selected_checkpoint_round(
    checkpoint_selection: str, configured_rounds: int, best_validation_round: int | None
) -> int:
    if checkpoint_selection == "final_round":
        return configured_rounds
    if checkpoint_selection == "best_validation" and best_validation_round is not None:
        return best_validation_round
    raise RuntimeError("checkpoint selection completed without a selectable communication round")


def train_distributed_federated(
    model,
    bundle,
    config: dict,
    run_dir: str | Path,
    context: ProcessContext,
    process_records: list[dict],
    resume_checkpoint: str | Path | None = None,
    stop_after_round: int | None = None,
    client_training=None,
) -> dict | None:
    """Execute synchronous FedAvg; only rank zero mutates shared run records."""

    path = Path(run_dir)
    checkpoint_dir = path / "checkpoints"
    logger = _logger(path) if context.is_coordinator else None
    torch.use_deterministic_algorithms(True, warn_only=True)
    model_initialization_id = state_identity(model.state_dict())
    initialization_artifact = {
        "model_initialization_id": model_initialization_id,
        "resolved_seed": bundle.resolved_seed_values["model_initialization"],
        "model_class": type(model).__name__,
        "source": "random_initialization",
        "centralized_checkpoint_used": False,
    }
    topology_mapping = [
        {
            "process_rank": value["rank"],
            "device_index": value["device_index"],
            "device_slot": value["device_slot"],
        }
        for value in process_records
    ]
    execution_identity = {
        "configuration_id": configuration_identity(config),
        "scientific_identity": distributed_scientific_identity(config),
        "configured_execution_identity": distributed_execution_identity(config),
        "git_commit": read_git_commit(path),
        "split_id": bundle.split_artifact["split_id"],
        "partition_id": bundle.partition.partition_id,
        "model_initialization_id": model_initialization_id,
        "resolved_seeds": bundle.resolved_seed_values,
        "dataset_identity": bundle.split_artifact.get("dataset_identity"),
        "node_count": config["parallel_execution"]["node_count"],
        "device_count": config["parallel_execution"]["device_count"],
        "process_count": config["parallel_execution"]["process_count"],
        "client_processes_per_device": config["parallel_execution"]["client_processes_per_device"],
        "control_backend": context.backend,
        "cuda_process_service": context.cuda_process_service,
        "client_assignment": config["parallel_execution"]["client_assignment"],
        "aggregation_order": config["parallel_execution"]["aggregation_order"],
        "process_to_device_mapping": topology_mapping,
        "hardware_allocation": {
            "visible_device_count": context.visible_device_count,
            "device_names": sorted({value["device_name"] for value in process_records if value["device_name"]}),
            "device_total_memory_bytes": sorted(
                {
                    value["device_total_memory_bytes"]
                    for value in process_records
                    if value["device_total_memory_bytes"] is not None
                }
            ),
        },
    }
    if context.is_coordinator:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _verify_or_write(path / "resolved_seeds.json", bundle.resolved_seed_values, "resolved seed identities")
        _verify_or_write(path / "split.json", bundle.split_artifact, "split identity")
        _verify_or_write(path / "partition.json", bundle.partition.artifact, "partition identity")
        _verify_or_write(
            path / "model_initialization.json",
            initialization_artifact,
            "model initialization identity",
        )
        _verify_or_write(path / "execution_provenance.json", execution_identity, "distributed execution identity")
        _verify_or_write(
            path / "process_mapping.json",
            {"processes": _stable_process_mapping(process_records)},
            "rank-to-device mapping",
        )
        measurements = _load_measurements(path, execution_identity)
        current_resource_allocation = {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "allocated_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "gpu_telemetry_path": os.environ.get("FEDAPFA_GPU_TELEMETRY"),
            "mps_log_archive": os.environ.get("FEDAPFA_MPS_ARCHIVE"),
            "mps_active_thread_percentage": os.environ.get("FEDAPFA_MPS_ACTIVE_THREAD_PERCENTAGE"),
        }
        measurements["resource_allocation"] = current_resource_allocation
        resource_allocations = list(measurements.get("resource_allocations", []))
        allocation_identity = (
            current_resource_allocation["array_job_id"],
            current_resource_allocation["array_task_id"],
            current_resource_allocation["job_id"],
        )
        if any(value is not None for value in allocation_identity) and not any(
            (
                value.get("array_job_id"),
                value.get("array_task_id"),
                value.get("job_id"),
            )
            == allocation_identity
            for value in resource_allocations
        ):
            resource_allocations.append(current_resource_allocation)
        measurements["resource_allocations"] = resource_allocations
        measurements["process_mapping"] = process_records
        process_mapping_attempts = list(measurements.get("process_mapping_attempts", []))
        process_mapping_attempts.append(
            {
                "execution_attempt": measurements["resume_count"] + 1,
                "processes": process_records,
            }
        )
        measurements["process_mapping_attempts"] = process_mapping_attempts
        atomic_write_json(path / "execution_measurements.json", measurements)
    else:
        measurements = None

    model.to(context.device)
    client_ids = bundle.client_ids
    if bundle.aggregation_weighting != config["federated"]["aggregation_weighting"]:
        raise RuntimeError("workload aggregation policy differs from the resolved configuration")
    if bundle.checkpoint_selection != config["federated"]["checkpoint_selection"]:
        raise RuntimeError("workload checkpoint policy differs from the resolved configuration")
    schedule = ClientSelectionSchedule(client_ids, bundle.resolved_seed_values["client_selection"])
    start_round = 1
    checkpoint_selection = config["federated"]["checkpoint_selection"]
    validation_enabled = checkpoint_selection == "best_validation"
    if context.is_coordinator and validation_enabled and bundle.validation_dataset is None:
        raise RuntimeError("best-validation selection requires coordinator validation data")
    if not context.is_coordinator and bundle.validation_dataset is not None:
        raise RuntimeError("non-coordinator processes must not construct validation data")
    best_accuracy: float | None = None
    best_round: int | None = None
    cumulative_download = 0
    cumulative_upload = 0
    client_records: list[dict] = []
    round_records: list[dict] = []
    if context.is_coordinator and resume_checkpoint is not None:
        checkpoint = load_federated_checkpoint(
            resume_checkpoint,
            model,
            config,
            path,
            bundle.split_artifact["split_id"],
            bundle.partition.partition_id,
            model_initialization_id,
        )
        start_round = checkpoint["next_round"]
        stored_best_accuracy = checkpoint.get("best_validation_accuracy")
        stored_best_round = checkpoint.get("best_validation_round")
        best_accuracy = None if stored_best_accuracy is None else float(stored_best_accuracy)
        best_round = None if stored_best_round is None else int(stored_best_round)
        cumulative_download = int(checkpoint["cumulative_download_bytes"])
        cumulative_upload = int(checkpoint["cumulative_upload_bytes"])
        client_records = list(checkpoint["client_records"])
        round_records = list(checkpoint["round_records"])
        schedule.load_state_dict(checkpoint["selection_generator_state"])
        if len(round_records) != start_round - 1:
            raise RuntimeError("checkpoint round records are incompatible with next_round")
        expected_clients = (start_round - 1) * config["federated"]["clients_per_round"]
        if len(client_records) != expected_clients:
            raise RuntimeError("checkpoint client records are incompatible with next_round")
        stored_round_path = path / "round_metrics.jsonl"
        if stored_round_path.is_file():
            stored_round_records = [
                json.loads(line)
                for line in stored_round_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(stored_round_records) == len(round_records) and [
                value.get("round_number") for value in stored_round_records
            ] == [value.get("round_number") for value in round_records]:
                round_records = stored_round_records
        _write_jsonl(path / "client_metrics.jsonl", client_records)
        _write_jsonl(path / "round_metrics.jsonl", round_records)
        logger.info("resuming distributed execution at communication round %d", start_round)

    control = [
        {
            "start_round": start_round,
            "best_accuracy": best_accuracy,
            "best_round": best_round,
            "cumulative_download": cumulative_download,
            "cumulative_upload": cumulative_upload,
        }
        if context.is_coordinator
        else None
    ]
    dist.broadcast_object_list(control, src=0, device=context.control_device)
    if not context.is_coordinator:
        start_round = int(control[0]["start_round"])

    rounds = config["federated"]["rounds"]
    payload = model_payload_bytes(model.state_dict())
    for round_number in range(start_round, rounds + 1):
        active_profiler = _start_round_profiler(config, context, path, round_number)
        _synchronize_processes(context)
        round_started = time.monotonic()
        selected_at_rank_zero = (
            schedule.select(round_number, config["federated"]["clients_per_round"])
            if context.is_coordinator
            else None
        )
        selected = broadcast_selected_clients(context, selected_at_rank_zero)
        assignments = assign_clients(selected, context.world_size)

        server_before = clone_state_dict(model.state_dict()) if context.is_coordinator else None
        synchronize_cuda(context.device)
        distribution_started = time.monotonic()
        with torch.profiler.record_function("global_model_distribution"):
            incoming_model_id = broadcast_model_state(model, context)
        _synchronize_processes(context)
        model_distribution_time = time.monotonic() - distribution_started

        _synchronize_processes(context)
        client_wall_started = time.monotonic()
        with torch.profiler.record_function("assigned_client_training"):
            rank_payload = train_rank_clients(
                model,
                bundle,
                config,
                context,
                assignments,
                round_number,
                incoming_model_id,
                payload,
                client_training,
            )
        _synchronize_processes(context)
        parallel_client_time = time.monotonic() - client_wall_started

        collection_started = time.monotonic()
        with torch.profiler.record_function("client_result_collection"):
            gathered_payloads, serialized_sizes = gather_rank_payloads(rank_payload, context)
        _synchronize_processes(context)
        result_collection_time = time.monotonic() - collection_started

        if context.is_coordinator:
            incoming_cpu = {name: value.detach().cpu() for name, value in server_before.items()}
            local_results, ordered_envelopes = order_client_results(
                gathered_payloads,
                assignments,
                round_number,
                config,
                incoming_cpu,
                incoming_model_id,
            )
            if any(not torch.equal(server_before[name], model.state_dict()[name]) for name in server_before):
                raise RuntimeError("coordinator model changed before FedAvg aggregation")

            synchronize_cuda(context.device)
            aggregation_started = time.monotonic()
            with torch.profiler.record_function("selected_order_aggregation"):
                weights, aggregated_update_norm, update_cosines = aggregate_client_results(
                    model,
                    local_results,
                    config["federated"]["aggregation_weighting"],
                )
            synchronize_cuda(context.device)
            aggregation_time = time.monotonic() - aggregation_started
            after_model_id = state_identity(model.state_dict())
            for envelope, weight, cosine in zip(ordered_envelopes, weights, update_cosines, strict=True):
                record = envelope.result.record(weight, cosine)
                record.update(
                    {
                        "selected_position": envelope.selected_position,
                        "process_rank": envelope.process_rank,
                        "device_index": process_records[envelope.process_rank]["device_index"],
                        "device_slot": process_records[envelope.process_rank]["device_slot"],
                        "incoming_global_model_id": envelope.incoming_global_model_id,
                        "update_identity": envelope.update_identity,
                    }
                )
                client_records.append(record)

            validation = None
            validation_time = 0.0
            improved = False
            if validation_enabled:
                validation_seed = derive_seed(config["seed"], config["seed_streams"]["validation"], round_number)
                synchronize_cuda(context.device)
                validation_started = time.monotonic()
                with torch.profiler.record_function("coordinator_validation"):
                    validation = validate_global_model(
                        model,
                        bundle.validation_dataset,
                        context.device,
                        config["federated"]["local_batch_size"],
                        validation_seed,
                        config["federated"]["data_loader_workers"],
                        config["federated"]["persistent_workers"],
                    )
                synchronize_cuda(context.device)
                validation_time = time.monotonic() - validation_started
                improved = best_accuracy is None or validation.accuracy > best_accuracy
                if improved:
                    best_accuracy = validation.accuracy
                    best_round = round_number

            communication = communication_for_clients(payload, len(selected))
            cumulative_download += communication["download_bytes"]
            cumulative_upload += communication["upload_bytes"]
            clients_by_rank, busy_by_rank, memory_by_rank, reserved_by_rank, examples_by_rank = (
                _round_process_records(gathered_payloads)
            )
            device_ids = range(config["parallel_execution"]["device_count"])
            clients_by_device = {
                str(device): [
                    client
                    for rank, clients in clients_by_rank.items()
                    if process_records[int(rank)]["device_index"] == device
                    for client in clients
                ]
                for device in device_ids
            }
            examples_by_device = {
                str(device): sum(
                    count
                    for rank, count in examples_by_rank.items()
                    if process_records[int(rank)]["device_index"] == device
                )
                for device in device_ids
            }
            busy_by_device = {
                str(device): sum(
                    duration
                    for rank, duration in busy_by_rank.items()
                    if process_records[int(rank)]["device_index"] == device
                )
                for device in device_ids
            }
            maximum_busy = max(busy_by_rank.values())
            process_load_imbalance = (
                (maximum_busy - min(busy_by_rank.values())) / maximum_busy if maximum_busy else 0.0
            )
            idle_by_process = {
                rank: max(0.0, parallel_client_time - duration) for rank, duration in busy_by_rank.items()
            }
            per_device_capacity = (
                parallel_client_time * config["parallel_execution"]["client_processes_per_device"]
            )
            idle_by_device = {
                device: max(0.0, per_device_capacity - duration)
                for device, duration in busy_by_device.items()
            }
            round_memory = [
                value
                for value in [
                    *memory_by_rank.values(),
                    None if validation is None else validation.peak_cuda_memory_bytes,
                ]
                if value is not None
            ]
            update_norms = [result.update_l2_norm for result in local_results]
            round_result = RoundResult(
                round_number=round_number,
                selected_client_ids=selected,
                aggregation_weighting=config["federated"]["aggregation_weighting"],
                client_example_counts=[result.example_count for result in local_results],
                client_training_examples_presented=[
                    result.local_training_examples_presented for result in local_results
                ],
                aggregation_weights=weights,
                total_selected_examples=sum(result.example_count for result in local_results),
                total_training_examples_presented=sum(
                    result.local_training_examples_presented for result in local_results
                ),
                validation_loss=None if validation is None else validation.loss,
                validation_accuracy=None if validation is None else validation.accuracy,
                validation_macro_f1=None if validation is None else validation.macro_f1,
                validation_per_class_accuracy=None if validation is None else validation.per_class_accuracy,
                validation_confusion_matrix=None if validation is None else validation.confusion_matrix,
                validation_spike_rates=None if validation is None else validation.spike_rates,
                global_model_l2_norm=global_model_norm(model),
                aggregated_update_l2_norm=aggregated_update_norm,
                mean_client_update_l2_norm=statistics.mean(update_norms),
                standard_deviation_client_update_l2_norm=statistics.pstdev(update_norms),
                mean_client_to_aggregate_cosine_similarity=statistics.mean(update_cosines),
                minimum_client_to_aggregate_cosine_similarity=min(update_cosines),
                maximum_client_to_aggregate_cosine_similarity=max(update_cosines),
                client_training_time_seconds=parallel_client_time,
                aggregation_time_seconds=aggregation_time,
                validation_time_seconds=validation_time,
                total_round_time_seconds=time.monotonic() - round_started,
                logical_download_bytes=communication["download_bytes"],
                logical_upload_bytes=communication["upload_bytes"],
                logical_communication_bytes=communication["total_bytes"],
                cumulative_logical_download_bytes=cumulative_download,
                cumulative_logical_upload_bytes=cumulative_upload,
                cumulative_logical_communication_bytes=cumulative_download + cumulative_upload,
                peak_cuda_memory_bytes=max(round_memory) if round_memory else None,
                current_best_validation_round=best_round,
                selected_checkpoint=(
                    improved if checkpoint_selection == "best_validation" else round_number == rounds
                ),
            ).record()
            round_result.update(
                {
                    "parallel_execution": config["parallel_execution"],
                    "selected_client_order": selected,
                    "client_assignments": _assignment_records(assignments, process_records),
                    "clients_by_process_rank": clients_by_rank,
                    "process_busy_time_seconds": busy_by_rank,
                    "examples_by_process_rank": examples_by_rank,
                    "clients_by_device_index": clients_by_device,
                    "examples_by_device_index": examples_by_device,
                    "combined_process_busy_time_seconds_by_device_index": busy_by_device,
                    "estimated_idle_time_seconds_by_process_rank": idle_by_process,
                    "estimated_idle_time_seconds_by_device_index": idle_by_device,
                    "process_load_imbalance": process_load_imbalance,
                    "client_training_durations_seconds": [value.execution_time_seconds for value in local_results],
                    "client_data_wait_durations_seconds": [value.data_wait_time_seconds for value in local_results],
                    "sum_client_training_durations_seconds": sum(
                        value.execution_time_seconds for value in local_results
                    ),
                    "model_distribution_time_seconds": model_distribution_time,
                    "parallel_client_training_wall_time_seconds": parallel_client_time,
                    "result_collection_time_seconds": result_collection_time,
                    "result_collection_bytes_by_process_rank": serialized_sizes,
                    "aggregation_order": "selected_client_order",
                    "ordered_update_identities": [value.update_identity for value in ordered_envelopes],
                    "global_model_identity_before_aggregation": incoming_model_id,
                    "global_model_identity_after_aggregation": after_model_id,
                    "execution_data_movement_bytes": payload * (context.world_size - 1)
                    + sum(serialized_sizes[1:]),
                    "checkpoint_time_seconds": None,
                }
            )
            round_records.append(round_result)
            _write_jsonl(path / "client_metrics.jsonl", client_records)
            _write_jsonl(path / "round_metrics.jsonl", round_records)
            checkpoint_arguments = {
                "model": model,
                "config": config,
                "run_dir": path,
                "next_round": round_number + 1,
                "best_validation_accuracy": best_accuracy,
                "best_validation_round": best_round,
                "selection_state": schedule.state_dict(),
                "split_id": bundle.split_artifact["split_id"],
                "partition_id": bundle.partition.partition_id,
                "model_initialization_id": model_initialization_id,
                "cumulative_download_bytes": cumulative_download,
                "cumulative_upload_bytes": cumulative_upload,
                "client_records": client_records,
                "round_records": round_records,
            }
            checkpoint_started = time.monotonic()
            with torch.profiler.record_function("coordinator_checkpoint_write"):
                if improved:
                    save_federated_checkpoint(checkpoint_dir / "best.pt", **checkpoint_arguments)
                save_federated_checkpoint(checkpoint_dir / "last.pt", **checkpoint_arguments)
            checkpoint_time = time.monotonic() - checkpoint_started
            round_result["checkpoint_time_seconds"] = checkpoint_time
            round_result["total_round_time_seconds"] = time.monotonic() - round_started
            _write_jsonl(path / "round_metrics.jsonl", round_records)
            measurement = {
                "round_number": round_number,
                "selected_client_order": selected,
                "physical_device_count": config["parallel_execution"]["device_count"],
                "client_processes_per_device": config["parallel_execution"]["client_processes_per_device"],
                "process_count": config["parallel_execution"]["process_count"],
                "process_to_device_mapping": process_records,
                "client_assignments": _assignment_records(assignments, process_records),
                "process_busy_time_seconds": busy_by_rank,
                "combined_process_busy_time_seconds_by_device_index": busy_by_device,
                "clients_by_process_rank": clients_by_rank,
                "examples_by_process_rank": examples_by_rank,
                "clients_by_device_index": clients_by_device,
                "examples_by_device_index": examples_by_device,
                "estimated_idle_time_seconds_by_process_rank": idle_by_process,
                "estimated_idle_time_seconds_by_device_index": idle_by_device,
                "process_load_imbalance": process_load_imbalance,
                "parallel_client_training_wall_time_seconds": parallel_client_time,
                "sum_client_training_durations_seconds": round_result[
                    "sum_client_training_durations_seconds"
                ],
                "model_distribution_time_seconds": model_distribution_time,
                "result_collection_time_seconds": result_collection_time,
                "aggregation_time_seconds": aggregation_time,
                "validation_time_seconds": validation_time,
                "checkpoint_time_seconds": checkpoint_time,
                "total_round_time_seconds": round_result["total_round_time_seconds"],
                "peak_cuda_memory_bytes_by_process_rank": memory_by_rank,
                "peak_cuda_reserved_bytes_by_process_rank": reserved_by_rank,
                "execution_data_movement_bytes": round_result["execution_data_movement_bytes"],
                "logical_communication_bytes": communication["total_bytes"],
                "selected_examples_per_second": round_result["total_training_examples_presented"]
                / parallel_client_time,
            }
            measurements["rounds"] = [
                value for value in measurements.get("rounds", []) if value.get("round_number") != round_number
            ]
            measurements["rounds"].append(measurement)
            measurements["rounds"].sort(key=lambda value: value["round_number"])
            atomic_write_json(path / "execution_measurements.json", measurements)
            if validation is None:
                logger.info(
                    "round=%d selected=%s validation=unavailable process_count=%d",
                    round_number,
                    ",".join(selected),
                    context.world_size,
                )
            else:
                logger.info(
                    "round=%d selected=%s validation_accuracy=%.6f best_round=%d process_count=%d",
                    round_number,
                    ",".join(selected),
                    validation.accuracy,
                    best_round,
                    context.world_size,
                )

        _stop_round_profiler(active_profiler)
        dist.barrier()
        if stop_after_round is not None and round_number >= stop_after_round:
            return (
                {
                    "completed": False,
                    "completed_rounds": round_number,
                    "selected_client_ids": [record["selected_client_ids"] for record in round_records],
                }
                if context.is_coordinator
                else None
            )

    if not context.is_coordinator:
        dist.barrier()
        return None
    if len(round_records) != rounds:
        raise RuntimeError("distributed execution ended before all communication rounds")
    if checkpoint_selection == "best_validation" and best_round is None:
        raise RuntimeError("best-validation selection completed without a validation result")
    if config["execution_measurement"]["record_device_utilization"]:
        utilization = _required_gpu_utilization_record(
            config, os.environ.get("FEDAPFA_GPU_TELEMETRY")
        )
        utilization["execution_attempt"] = measurements["resume_count"] + 1
        utilization["scope"] = "latest_execution_attempt"
        measurements["gpu_utilization"] = utilization
        atomic_write_json(path / "execution_measurements.json", measurements)
    selected_round = _selected_checkpoint_round(checkpoint_selection, rounds, best_round)
    selected_checkpoint_path = (
        checkpoint_dir / "best.pt" if checkpoint_selection == "best_validation" else checkpoint_dir / "last.pt"
    )
    load_federated_checkpoint(
        selected_checkpoint_path,
        model,
        config,
        path,
        bundle.split_artifact["split_id"],
        bundle.partition.partition_id,
        model_initialization_id,
        restore_random_states=False,
    )
    official_path = path / "official_test_metrics.json"
    official_identity = {
        "selected_round": selected_round,
        "checkpoint_selection": checkpoint_selection,
        "split_id": bundle.split_artifact["split_id"],
        "partition_id": bundle.partition.partition_id,
        "model_initialization_id": model_initialization_id,
    }
    official_record = _load_official_test_record(official_path, official_identity)
    official_started = time.monotonic()
    if official_record is None:
        atomic_write_json(
            official_path,
            {
                **official_identity,
                "access_count": 1,
                "monitored_during_rounds": False,
                "evaluated_after_model_selection": True,
                "evaluation_completed": False,
                "complete_split": None,
                "metrics": None,
                "dataset_identity": None,
            },
        )
        test_dataset = bundle.official_test_dataset(model_selected=True)
        test_result = evaluate_model(
            model,
            test_dataset,
            context.device,
            config["federated"]["local_batch_size"],
            bundle.resolved_seed_values["final_test"],
            config["federated"]["data_loader_workers"],
            config["federated"]["persistent_workers"],
        )
        official_test_time = time.monotonic() - official_started
        official_record = {
            **official_identity,
            "access_count": 1,
            "monitored_during_rounds": False,
            "evaluated_after_model_selection": True,
            "evaluation_completed": True,
            "complete_split": True,
            "metrics": test_result.__dict__,
            "dataset_identity": getattr(bundle, "official_test_identity", None),
            "evaluation_time_seconds": official_test_time,
        }
        atomic_write_json(official_path, official_record)
    measurements["official_test_time_seconds"] = official_record.get(
        "evaluation_time_seconds", time.monotonic() - official_started
    )
    measurements["completed"] = True
    measurements["completed_rounds"] = rounds
    measurements["total_round_time_seconds"] = sum(
        record["total_round_time_seconds"] for record in measurements["rounds"]
    )
    atomic_write_json(path / "execution_measurements.json", measurements)

    selected_record = round_records[selected_round - 1]
    selected_validation = None
    fairness_proxy = None
    if validation_enabled:
        selected_validation = {
            "loss": selected_record["validation_loss"],
            "accuracy": selected_record["validation_accuracy"],
            "macro_f1": selected_record["validation_macro_f1"],
            "per_class_accuracy": selected_record["validation_per_class_accuracy"],
            "confusion_matrix": selected_record["validation_confusion_matrix"],
            "spike_rates": selected_record["validation_spike_rates"],
        }
        fairness_proxy = fairness_proxy_record(
            selected_record["validation_per_class_accuracy"], bundle.partition.artifact
        )
    test_metrics = official_record["metrics"]
    training_example_count = len(bundle.split_artifact["training_indices"])
    validation_example_count = len(bundle.split_artifact["validation_indices"])
    total_execution_data_movement = sum(
        record["execution_data_movement_bytes"] for record in round_records
    )
    final = {
        "schema_version": 2,
        "accepted": False,
        "completed": False,
        "completed_rounds": rounds,
        "best_validation_accuracy": best_accuracy,
        "selected_round": selected_round,
        "checkpoint_selection": checkpoint_selection,
        "final_validation_accuracy": round_records[-1]["validation_accuracy"],
        "selected_validation": selected_validation,
        "client_distribution_weighted_validation_accuracy": fairness_proxy,
        "test": test_metrics,
        "model_class": type(model).__name__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "configuration_id": configuration_identity(config),
        "scientific_identity": distributed_scientific_identity(config),
        "execution_identity": execution_identity,
        "split_id": bundle.split_artifact["split_id"],
        "partition_id": bundle.partition.partition_id,
        "model_initialization_id": model_initialization_id,
        "resolved_seeds": bundle.resolved_seed_values,
        "logical_communication": {
            "definition": "communicated tensor element count multiplied by element size",
            "model_payload_bytes": payload,
            "cumulative_download_bytes": cumulative_download,
            "cumulative_upload_bytes": cumulative_upload,
            "cumulative_total_bytes": cumulative_download + cumulative_upload,
            "optimizer_state_included": False,
            "dataset_transfer_included": False,
            "checkpoint_io_included": False,
            "telemetry_files_included": False,
            "measured_network_traffic": False,
        },
        "execution_data_movement": {
            "definition": "model-state and result movement between processes on one node",
            "total_bytes": total_execution_data_movement,
            "included_in_logical_federated_communication": False,
        },
        "parallel_execution": {
            **config["parallel_execution"],
            "process_mapping": process_records,
        },
        "execution_time_seconds": sum(record["total_round_time_seconds"] for record in round_records),
        "mean_client_update_l2_norm": sum(record["update_l2_norm"] for record in client_records)
        / len(client_records),
        "mean_client_spike_rates": _mean_client_spike_rates(client_records),
        "mean_client_update_cosine_similarity": statistics.mean(
            record["update_cosine_similarity"] for record in client_records
        ),
        "peak_cuda_memory_bytes": (
            max(
                value
                for value in (record["peak_cuda_memory_bytes"] for record in round_records)
                if value is not None
            )
            if any(record["peak_cuda_memory_bytes"] is not None for record in round_records)
            else None
        ),
        "dataset_identity": bundle.split_artifact.get("dataset_identity"),
        "data_protocol": {
            "examples_available_before_validation_separation": training_example_count
            + (
                validation_example_count
                if bundle.evaluation_protocol["internal_validation_available"]
                else 0
            ),
            "examples_used_for_client_training": training_example_count,
            "examples_used_for_validation": validation_example_count,
            "official_test_examples": int(test_metrics["examples"]),
            "official_test_access_count": int(official_record["access_count"]),
            "selected_checkpoint_rule": checkpoint_selection,
            "validation_collection": bundle.evaluation_protocol["validation_collection"],
            "internal_validation_available": bundle.evaluation_protocol[
                "internal_validation_available"
            ],
            "official_test_monitored_during_training": False,
            "official_test_paper_collection_name": bundle.evaluation_protocol[
                "official_test_publication_collection_name"
            ],
            "released_source_monitors_official_test_during_training": bundle.evaluation_protocol[
                "external_implementation_monitors_official_test"
            ],
            "complete_standard_training_collection_used": bundle.evaluation_protocol[
                "complete_standard_training_collection"
            ],
            "all_50000_standard_training_examples_used": (
                bundle.evaluation_protocol["complete_standard_training_collection"]
                and training_example_count == 50000
            ),
        },
        "data_loader": {
            "num_workers": config["federated"]["data_loader_workers"],
            "persistent_workers": config["federated"]["persistent_workers"],
            "pin_memory": config["federated"].get("pin_memory", False),
            "prefetch_factor": config["federated"].get("prefetch_factor"),
            "non_blocking_cuda_transfer": config["federated"].get("non_blocking_transfer", True),
        },
        "execution_measurement": config["execution_measurement"],
        "selected_checkpoint_artifact": (
            "checkpoints/best.pt" if checkpoint_selection == "best_validation" else "checkpoints/last.pt"
        ),
        "aggregation_weighting": config["federated"]["aggregation_weighting"],
        "local_epochs": config["federated"]["local_epochs"],
        "total_clients": config["federated"]["clients"],
        "participating_clients": config["federated"]["clients_per_round"],
        "momentum": config["federated"].get("momentum"),
        "weight_decay": config["federated"]["weight_decay"],
        "distribution": config["federated"]["partition"]["method"],
        "partition_alpha": config["federated"]["partition"].get("alpha"),
        "timesteps": config["model"].get("timesteps"),
        "input_encoding": config["model"].get("input_encoding"),
        "protocol_assumptions": config.get("protocol_assumptions", []),
        "termination": {"reason": "communication_rounds_completed", "configured_rounds": rounds},
    }
    atomic_write_json(path / "final_metrics.json", final)
    acceptance = evaluate_federated_acceptance(config, path, final)
    distributed_failures = _verify_distributed_records(config, round_records, measurements)
    if distributed_failures:
        acceptance["completion_failures"].extend(distributed_failures)
        acceptance["accepted"] = False
        acceptance["completed"] = False
    acceptance["parallel_execution"] = config["parallel_execution"]
    acceptance["execution_provenance"] = execution_identity
    final["accepted"] = acceptance["completed"]
    final["completed"] = acceptance["completed"]
    final["scientific_status"] = acceptance["scientific_status"]
    atomic_write_json(path / "final_metrics.json", final)
    atomic_write_json(path / "acceptance.json", acceptance)
    logger.info("finished completed=%s scientific_status=%s", final["completed"], final["scientific_status"])
    reset_snn_state(model)
    dist.barrier()
    return final
