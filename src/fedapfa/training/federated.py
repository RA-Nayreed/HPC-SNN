"""Single-device orchestration for the SHD FedAvg scientific reference."""

from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from fedapfa.federated.acceptance import evaluate_federated_acceptance
from fedapfa.federated.aggregation import clone_state_dict
from fedapfa.federated.checkpointing import (
    configuration_identity,
    load_federated_checkpoint,
    save_federated_checkpoint,
    state_identity,
)
from fedapfa.federated.client import evaluate_model, reset_snn_state, synchronize_cuda, train_client
from fedapfa.federated.client_sampling import ClientSelectionSchedule
from fedapfa.federated.communication_accounting import communication_for_clients, model_payload_bytes
from fedapfa.federated.fedavg import aggregate_client_results
from fedapfa.federated.randomness import derive_seed, resolved_seeds
from fedapfa.federated.round_state import RoundResult
from fedapfa.federated.server import global_model_norm, validate_global_model
from fedapfa.models.model_factory import make_model
from fedapfa.training.centralized import resolve_device
from fedapfa.utilities.serialization import atomic_write_json, atomic_write_text


def make_initialized_federated_model(config: dict):
    """Create the global model from its isolated deterministic stream."""

    seeds = resolved_seeds(config)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    seed = seeds["model_initialization"]
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        model = make_model(config)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
    return model


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"fedapfa.federated.{path.name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(path / "training.log", mode="a", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _write_jsonl(path: Path, records: list[dict]) -> None:
    atomic_write_text(path, "".join(json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records))


def _verify_or_write(path: Path, value: dict, identity_key: str | None = None) -> None:
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != value:
            label = identity_key or path.name
            raise RuntimeError(f"stored {label} is incompatible with the resolved execution")
    else:
        atomic_write_json(path, value)


def _mean_client_spike_rates(records: list[dict]) -> dict[str, float]:
    names = sorted({name for record in records for name in record.get("spike_rates", {})})
    return {
        name: sum(record["spike_rates"][name] for record in records) / len(records)
        for name in names
        if all(name in record.get("spike_rates", {}) for record in records)
    }


def _load_official_test_record(path: Path, expected: dict) -> dict | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(f"official test record {key} is incompatible")
    if value.get("access_count") != 1:
        raise RuntimeError("official test record must contain exactly one access")
    if value.get("evaluation_completed") is not True:
        raise RuntimeError(
            "an official test evaluation started without a completed durable record; automatic reevaluation is refused"
        )
    return value


def train_federated(
    model,
    bundle,
    config: dict,
    run_dir: str | Path,
    resume_checkpoint: str | Path | None = None,
    stop_after_round: int | None = None,
) -> dict:
    """Execute or resume FedAvg while keeping official-test access after selection."""

    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = _logger(path)
    device = resolve_device(config["device"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    model_initialization_id = state_identity(model.state_dict())
    initialization_artifact = {
        "model_initialization_id": model_initialization_id,
        "resolved_seed": bundle.resolved_seed_values["model_initialization"],
        "model_class": type(model).__name__,
        "source": "random_initialization",
        "centralized_checkpoint_used": False,
    }
    _verify_or_write(path / "resolved_seeds.json", bundle.resolved_seed_values, "resolved seed identities")
    _verify_or_write(path / "split.json", bundle.split_artifact, "split identity")
    _verify_or_write(path / "partition.json", bundle.partition.artifact, "partition identity")
    _verify_or_write(path / "model_initialization.json", initialization_artifact, "model initialization identity")

    model.to(device)
    client_ids = sorted(bundle.partition.client_indices)
    schedule = ClientSelectionSchedule(client_ids, bundle.resolved_seed_values["client_selection"])
    start_round = 1
    best_accuracy = -math.inf
    best_round = 0
    cumulative_download = 0
    cumulative_upload = 0
    client_records: list[dict] = []
    round_records: list[dict] = []
    if resume_checkpoint is not None:
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
        best_accuracy = float(checkpoint["best_validation_accuracy"])
        best_round = int(checkpoint["best_validation_round"])
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
        _write_jsonl(path / "client_metrics.jsonl", client_records)
        _write_jsonl(path / "round_metrics.jsonl", round_records)
        logger.info("resuming at communication round %d", start_round)

    rounds = config["federated"]["rounds"]
    payload = model_payload_bytes(model.state_dict())
    for round_number in range(start_round, rounds + 1):
        synchronize_cuda(device)
        round_started = time.monotonic()
        selected = schedule.select(round_number, config["federated"]["clients_per_round"])
        server_before = clone_state_dict(model.state_dict())
        local_results = []
        client_training_started = time.monotonic()
        for client_id in selected:
            training_seed = derive_seed(
                config["seed"],
                config["seed_streams"]["client_training"],
                round_number,
                client_id,
            )
            result = train_client(
                model,
                bundle.client_dataset(client_id),
                client_id,
                round_number,
                config,
                device,
                training_seed,
                payload,
            )
            local_results.append(result)
        synchronize_cuda(device)
        client_training_time = time.monotonic() - client_training_started
        if any(not torch.equal(server_before[name], model.state_dict()[name]) for name in server_before):
            raise RuntimeError("server model changed before FedAvg aggregation")

        synchronize_cuda(device)
        aggregation_started = time.monotonic()
        weights, aggregated_update_norm = aggregate_client_results(model, local_results)
        synchronize_cuda(device)
        aggregation_time = time.monotonic() - aggregation_started
        for result, weight in zip(local_results, weights, strict=True):
            client_records.append(result.record(weight))

        validation_seed = derive_seed(
            config["seed"], config["seed_streams"]["validation"], round_number
        )
        synchronize_cuda(device)
        validation_started = time.monotonic()
        validation = validate_global_model(
            model,
            bundle.validation_dataset,
            device,
            config["federated"]["local_batch_size"],
            validation_seed,
            config["federated"]["data_loader_workers"],
            config["federated"]["persistent_workers"],
        )
        synchronize_cuda(device)
        validation_time = time.monotonic() - validation_started
        improved = validation.accuracy > best_accuracy
        if improved:
            best_accuracy = validation.accuracy
            best_round = round_number

        communication = communication_for_clients(payload, len(selected))
        cumulative_download += communication["download_bytes"]
        cumulative_upload += communication["upload_bytes"]
        synchronize_cuda(device)
        total_round_time = time.monotonic() - round_started
        round_result = RoundResult(
            round_number=round_number,
            selected_client_ids=selected,
            client_example_counts=[result.example_count for result in local_results],
            aggregation_weights=weights,
            total_selected_examples=sum(result.example_count for result in local_results),
            validation_loss=validation.loss,
            validation_accuracy=validation.accuracy,
            validation_spike_rates=validation.spike_rates,
            global_model_l2_norm=global_model_norm(model),
            aggregated_update_l2_norm=aggregated_update_norm,
            client_training_time_seconds=client_training_time,
            aggregation_time_seconds=aggregation_time,
            validation_time_seconds=validation_time,
            total_round_time_seconds=total_round_time,
            logical_download_bytes=communication["download_bytes"],
            logical_upload_bytes=communication["upload_bytes"],
            logical_communication_bytes=communication["total_bytes"],
            cumulative_logical_download_bytes=cumulative_download,
            cumulative_logical_upload_bytes=cumulative_upload,
            cumulative_logical_communication_bytes=cumulative_download + cumulative_upload,
            current_best_validation_round=best_round,
            selected_checkpoint=improved,
        )
        round_records.append(round_result.record())
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
        if improved:
            save_federated_checkpoint(checkpoint_dir / "best.pt", **checkpoint_arguments)
        save_federated_checkpoint(checkpoint_dir / "last.pt", **checkpoint_arguments)
        logger.info(
            "round=%d selected=%s validation_accuracy=%.6f best_round=%d",
            round_number,
            ",".join(selected),
            validation.accuracy,
            best_round,
        )
        if stop_after_round is not None and round_number >= stop_after_round:
            return {
                "completed": False,
                "completed_rounds": round_number,
                "selected_client_ids": [record["selected_client_ids"] for record in round_records],
            }

    if len(round_records) != rounds:
        raise RuntimeError("federated execution ended before all communication rounds")
    best_checkpoint = checkpoint_dir / "best.pt"
    load_federated_checkpoint(
        best_checkpoint,
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
        "selected_round": best_round,
        "split_id": bundle.split_artifact["split_id"],
        "partition_id": bundle.partition.partition_id,
        "model_initialization_id": model_initialization_id,
    }
    official_record = _load_official_test_record(official_path, official_identity)
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
            },
        )
        test_dataset = bundle.official_test_dataset(model_selected=True)
        final_test = evaluate_model(
            model,
            test_dataset,
            device,
            config["federated"]["local_batch_size"],
            bundle.resolved_seed_values["final_test"],
            config["federated"]["data_loader_workers"],
            config["federated"]["persistent_workers"],
        )
        official_record = {
            **official_identity,
            "access_count": 1,
            "monitored_during_rounds": False,
            "evaluated_after_model_selection": True,
            "evaluation_completed": True,
            "complete_split": True,
            "metrics": final_test.__dict__,
        }
        atomic_write_json(official_path, official_record)
    test_metrics = official_record["metrics"]
    final = {
        "schema_version": 1,
        "accepted": False,
        "completed": False,
        "completed_rounds": rounds,
        "best_validation_accuracy": best_accuracy,
        "selected_round": best_round,
        "final_validation_accuracy": round_records[-1]["validation_accuracy"],
        "test": test_metrics,
        "model_class": type(model).__name__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "configuration_id": configuration_identity(config),
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
        "execution_time_seconds": sum(record["total_round_time_seconds"] for record in round_records),
        "mean_client_update_l2_norm": sum(record["update_l2_norm"] for record in client_records)
        / len(client_records),
        "mean_client_spike_rates": _mean_client_spike_rates(client_records),
        "termination": {"reason": "communication_rounds_completed", "configured_rounds": rounds},
    }
    atomic_write_json(path / "final_metrics.json", final)
    acceptance = evaluate_federated_acceptance(config, path, final)
    final["accepted"] = acceptance["completed"]
    final["completed"] = acceptance["completed"]
    final["scientific_status"] = acceptance["scientific_status"]
    atomic_write_json(path / "final_metrics.json", final)
    atomic_write_json(path / "acceptance.json", acceptance)
    logger.info("finished completed=%s scientific_status=%s", final["completed"], final["scientific_status"])
    reset_snn_state(model)
    return final
