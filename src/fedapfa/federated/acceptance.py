"""Truthful completion assessment for federated scientific executions."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fedapfa.utilities.serialization import sha256_json

from .checkpointing import configuration_identity


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"{path} contains a non-object record")
    return records


def evaluate_federated_acceptance(config: dict, run_dir: str | Path, final_metrics: dict) -> dict:
    path = Path(run_dir)
    failures: list[str] = []
    try:
        partition = _read_json(path / "partition.json")
        split = _read_json(path / "split.json")
        initialization = _read_json(path / "model_initialization.json")
        official_test = _read_json(path / "official_test_metrics.json")
        client_records = _read_jsonl(path / "client_metrics.jsonl")
        round_records = _read_jsonl(path / "round_metrics.jsonl")
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        partition = split = initialization = official_test = {}
        client_records = round_records = []
        failures.append(f"required scientific record is missing or invalid: {error}")
    rounds = config["federated"]["rounds"]
    clients_per_round = config["federated"]["clients_per_round"]
    validation_expected = config["federated"]["checkpoint_selection"] == "best_validation"
    if len(round_records) != rounds:
        failures.append(f"expected {rounds} round records, found {len(round_records)}")
    if len(client_records) != rounds * clients_per_round:
        failures.append("client record count does not match rounds and participation")
    if not _finite_tree(round_records) or not _finite_tree(client_records) or not _finite_tree(final_metrics):
        failures.append("federated metrics contain NaN or infinity")
    expected_rounds = list(range(1, rounds + 1))
    if [record.get("round_number") for record in round_records] != expected_rounds:
        failures.append("round records are missing, duplicated, or out of order")
    client_required = {
        "round_number",
        "client_id",
        "example_count",
        "client_population_examples",
        "presented_examples_per_local_epoch",
        "local_training_examples_presented",
        "batch_count",
        "starting_training_loss",
        "starting_training_accuracy",
        "ending_training_loss",
        "ending_training_accuracy",
        "spike_rates",
        "execution_time_seconds",
        "data_wait_time_seconds",
        "update_l2_norm",
        "peak_cuda_memory_bytes",
        "peak_cuda_reserved_bytes",
        "logical_download_bytes",
        "logical_upload_bytes",
        "logical_total_bytes",
        "aggregation_weight",
        "resolved_training_seed",
        "resolved_learning_rate",
        "initial_local_loss",
        "initial_local_accuracy",
        "final_local_loss",
        "final_local_accuracy",
        "client_spike_rates",
        "training_duration_seconds",
        "update_cosine_similarity",
    }
    round_required = {
        "round_number",
        "selected_client_ids",
        "aggregation_weighting",
        "client_example_counts",
        "client_training_examples_presented",
        "aggregation_weights",
        "total_selected_examples",
        "total_training_examples_presented",
        "validation_loss",
        "validation_accuracy",
        "validation_macro_f1",
        "validation_per_class_accuracy",
        "validation_confusion_matrix",
        "validation_spike_rates",
        "global_model_l2_norm",
        "aggregated_update_l2_norm",
        "mean_client_update_l2_norm",
        "standard_deviation_client_update_l2_norm",
        "mean_client_to_aggregate_cosine_similarity",
        "minimum_client_to_aggregate_cosine_similarity",
        "maximum_client_to_aggregate_cosine_similarity",
        "client_training_time_seconds",
        "aggregation_time_seconds",
        "validation_time_seconds",
        "total_round_time_seconds",
        "logical_download_bytes",
        "logical_upload_bytes",
        "logical_communication_bytes",
        "cumulative_logical_download_bytes",
        "cumulative_logical_upload_bytes",
        "cumulative_logical_communication_bytes",
        "peak_cuda_memory_bytes",
        "current_best_validation_round",
        "selected_checkpoint",
    }
    for index, record in enumerate(client_records):
        missing = client_required.difference(record)
        if missing:
            failures.append(f"client record {index} is missing fields: {sorted(missing)}")
        if not isinstance(record.get("spike_rates"), dict) or not record.get("spike_rates"):
            failures.append(f"client record {index} has no spike rates")
        if config["device"] == "cuda" and not isinstance(record.get("peak_cuda_memory_bytes"), int):
            failures.append(f"client record {index} has no CUDA-memory measurement")
        if record.get("example_count") != record.get("client_population_examples"):
            failures.append(f"client record {index} does not preserve its population size")
    for record in round_records:
        missing = round_required.difference(record)
        if missing:
            failures.append(f"round {record.get('round_number')} is missing fields: {sorted(missing)}")
        selected = record.get("selected_client_ids", [])
        weights = record.get("aggregation_weights", [])
        if len(selected) != clients_per_round or len(selected) != len(set(selected)):
            failures.append(f"round {record.get('round_number')} has an invalid client selection")
        if (
            len(weights) != clients_per_round
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in weights)
            or not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12)
        ):
            failures.append(f"round {record.get('round_number')} has invalid aggregation weights")
        policy = config["federated"]["aggregation_weighting"]
        if record.get("aggregation_weighting") != policy:
            failures.append(f"round {record.get('round_number')} has the wrong aggregation policy")
        if policy == "uniform" and any(value != 1.0 / clients_per_round for value in weights):
            failures.append(f"round {record.get('round_number')} does not use exact uniform weights")
        if policy == "example_count" and record.get("client_example_counts"):
            counts = record["client_example_counts"]
            expected_weights = [count / sum(counts) for count in counts]
            if weights != expected_weights:
                failures.append(f"round {record.get('round_number')} does not use exact example-count weights")
        validation_fields = (
            "validation_loss",
            "validation_accuracy",
            "validation_macro_f1",
            "validation_per_class_accuracy",
            "validation_confusion_matrix",
            "validation_spike_rates",
            "current_best_validation_round",
        )
        if validation_expected:
            if not isinstance(record.get("validation_spike_rates"), dict) or not record.get("validation_spike_rates"):
                failures.append(f"round {record.get('round_number')} has no validation spike rates")
        elif any(record.get(field) is not None for field in validation_fields):
            failures.append(f"round {record.get('round_number')} invents unavailable validation metrics")
        if not validation_expected and record.get("validation_time_seconds") != 0.0:
            failures.append(f"round {record.get('round_number')} has validation time without validation data")
        if record.get("logical_communication_bytes") != record.get("logical_download_bytes", 0) + record.get(
            "logical_upload_bytes", 0
        ):
            failures.append(f"round {record.get('round_number')} has inconsistent communication accounting")
        if record.get("cumulative_logical_communication_bytes") != record.get(
            "cumulative_logical_download_bytes", 0
        ) + record.get("cumulative_logical_upload_bytes", 0):
            failures.append(f"round {record.get('round_number')} has inconsistent cumulative communication")
    if partition.get("client_count") != config["federated"]["clients"]:
        failures.append("partition does not contain the expected clients")
    integrity = partition.get("integrity_checks", {})
    if not isinstance(integrity, dict) or not integrity or not all(integrity.values()):
        failures.append("partition integrity was not verified")
    partition_core = dict(partition)
    recorded_partition_id = partition_core.pop("partition_id", None)
    if recorded_partition_id != sha256_json(partition_core):
        failures.append("partition identity does not match its artifact")
    assigned = [index for client in partition.get("clients", []) for index in client.get("indices", [])]
    training_indices = split.get("training_indices", [])
    validation_indices = split.get("validation_indices", [])
    split_core = dict(split)
    recorded_split_id = split_core.pop("split_id", None)
    if recorded_split_id != sha256_json(split_core):
        failures.append("split identity does not match its artifact")
    if sorted(assigned) != sorted(training_indices) or len(assigned) != len(set(assigned)):
        failures.append("eligible training indices were not assigned exactly once")
    if split.get("validation_collection") != "official_validation" and set(assigned).intersection(validation_indices):
        failures.append("validation indices leaked into client partitions")
    minimum_size = config["federated"]["partition"]["minimum_examples_per_client"]
    if any(client.get("size") != len(client.get("indices", [])) for client in partition.get("clients", [])):
        failures.append("partition client sizes do not match index lists")
    if any(client.get("size", 0) < minimum_size for client in partition.get("clients", [])):
        failures.append("partition contains a client below the configured minimum size")
    checkpoint_names = ["last.pt"]
    if config["federated"]["checkpoint_selection"] == "best_validation":
        checkpoint_names.append("best.pt")
    for checkpoint_name in checkpoint_names:
        checkpoint = path / "checkpoints" / checkpoint_name
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            failures.append(f"missing checkpoint: checkpoints/{checkpoint_name}")
    selected_records = [record for record in round_records if record.get("selected_checkpoint") is True]
    if config["federated"]["checkpoint_selection"] == "final_round":
        if [record.get("round_number") for record in selected_records] != [rounds]:
            failures.append("final-round selection must identify only the configured final round")
    elif not selected_records:
        failures.append("best-validation selection did not identify a checkpoint")
    for log_name in ("training.log", "client_metrics.jsonl", "round_metrics.jsonl"):
        log = path / log_name
        if not log.is_file() or log.stat().st_size == 0:
            failures.append(f"{log_name} is missing or empty")
    if official_test.get("access_count") != 1:
        failures.append("official test evaluation must exist exactly once")
    if official_test.get("complete_split") is not True:
        failures.append("official test did not use the complete configured split")
    if official_test.get("evaluation_completed") is not True:
        failures.append("official test record is incomplete")
    if official_test.get("evaluated_after_model_selection") is not True:
        failures.append("official test was not evaluated after model selection")
    if official_test.get("monitored_during_rounds") is not False:
        failures.append("official test was accessed during communication rounds")
    if not final_metrics.get("test"):
        failures.append("official test metrics are missing")
    else:
        required_test_metrics = {
            "loss",
            "accuracy",
            "macro_f1",
            "per_class_accuracy",
            "confusion_matrix",
            "spike_rates",
        }
        missing_test_metrics = required_test_metrics.difference(final_metrics["test"])
        if missing_test_metrics:
            failures.append(f"official test metrics are missing fields: {sorted(missing_test_metrics)}")
    if any(
        config.get("subset", {}).get(key) != 0 for key in ("train_examples", "validation_examples", "test_examples")
    ):
        failures.append("scientific execution used configured sample caps")
    if any(
        config.get("training", {}).get(key) is not None
        for key in ("max_train_batches", "max_validation_batches", "max_test_batches")
    ):
        failures.append("scientific execution used configured batch caps")
    if not split.get("dataset_identity") and not final_metrics.get("dataset_identity"):
        failures.append("dataset identity is missing")
    for identity_name, artifact, key in (
        ("configuration", final_metrics, "configuration_id"),
        ("partition", partition, "partition_id"),
        ("split", split, "split_id"),
        ("model initialization", initialization, "model_initialization_id"),
    ):
        if not artifact.get(key):
            failures.append(f"{identity_name} identity is missing")
    if final_metrics.get("configuration_id") != configuration_identity(config):
        failures.append("configuration identity does not match the resolved configuration")
    git_commit = None
    try:
        git_commit = _read_json(path / "git.json").get("commit")
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        pass
    if not git_commit:
        failures.append("Git provenance is missing")
    communication = final_metrics.get("logical_communication", {})
    if (
        not isinstance(communication, dict)
        or communication.get("model_payload_bytes", 0) <= 0
        or communication.get("cumulative_total_bytes", 0) <= 0
        or communication.get("cumulative_total_bytes")
        != communication.get("cumulative_download_bytes", 0) + communication.get("cumulative_upload_bytes", 0)
    ):
        failures.append("logical communication accounting is missing or inconsistent")
    if final_metrics.get("completed_rounds") != rounds:
        failures.append("not all configured communication rounds completed")
    if final_metrics.get("model_class") != config["acceptance"]["expected_model_class"]:
        failures.append("model class does not match the configured reference")
    if validation_expected:
        if final_metrics.get("best_validation_accuracy") is None or final_metrics.get("selected_validation") is None:
            failures.append("checkpoint-selection validation metrics are missing")
    else:
        unavailable = (
            final_metrics.get("best_validation_accuracy"),
            final_metrics.get("final_validation_accuracy"),
            final_metrics.get("selected_validation"),
            final_metrics.get("client_distribution_weighted_validation_accuracy"),
        )
        if any(value is not None for value in unavailable):
            failures.append("final record invents unavailable internal-validation metrics")
        if validation_indices:
            failures.append("zero-validation execution contains internal validation indices")
    if (
        config["federated"]["checkpoint_selection"] == "final_round"
        and final_metrics.get("selected_round") != rounds
    ):
        failures.append("final-round checkpoint selection did not select the configured final round")

    completed = not failures
    achieved = final_metrics.get("test", {}).get("accuracy")
    reference = config["acceptance"].get("reference_test_accuracy")
    tolerance = config["acceptance"].get("absolute_tolerance")
    difference = None if reference is None or achieved is None else abs(float(achieved) - float(reference))
    descriptive_reference = config["acceptance"].get("descriptive_reference_accuracy")
    descriptive_signed_difference = (
        None
        if descriptive_reference is None or achieved is None
        else (float(achieved) - float(descriptive_reference)) * 100
    )
    descriptive_absolute_difference = (
        None if descriptive_signed_difference is None else abs(descriptive_signed_difference)
    )
    if reference is not None:
        scientific_status = "passed" if completed and difference is not None and difference <= tolerance else "failed"
    elif config["dataset"]["name"] == "cifar10":
        scientific_status = "equivalence_not_established"
    else:
        scientific_status = "not_claimed"
    return {
        "mode": config["mode"],
        "accepted": completed,
        "completed": completed,
        "completion_failures": failures,
        "scientific_status": scientific_status,
        "reference_test_accuracy": reference,
        "descriptive_reference_accuracy": descriptive_reference,
        "signed_descriptive_accuracy_difference_percentage_points": descriptive_signed_difference,
        "absolute_descriptive_accuracy_difference_percentage_points": descriptive_absolute_difference,
        "achieved_test_accuracy": achieved,
        "absolute_accuracy_difference": difference,
        "tolerance": tolerance,
        "protocol": config["protocol"],
        "seed": config["seed"],
        "git_commit": git_commit,
        "dataset": config["dataset"]["name"],
        "model_class": final_metrics.get("model_class"),
        "partition_id": partition.get("partition_id"),
        "split_id": split.get("split_id"),
        "model_initialization_id": initialization.get("model_initialization_id"),
        "dataset_identity": split.get("dataset_identity") or final_metrics.get("dataset_identity"),
        "protocol_assumptions": config.get("protocol_assumptions", []),
        "client_fairness_proxy_definition": (
            "This is a distribution-weighted proxy, not observed accuracy on private client test data."
        ),
        "official_test_access_information": {
            "access_count": official_test.get("access_count"),
            "monitored_during_rounds": official_test.get("monitored_during_rounds"),
            "evaluated_after_model_selection": official_test.get("evaluated_after_model_selection"),
            "complete_split": official_test.get("complete_split"),
        },
    }
