"""Strict validation for the single-GPU SHD FedAvg reference."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .validation import ConfigurationError

REQUIRED_STREAMS = {
    "split",
    "partition",
    "model_initialization",
    "client_selection",
    "client_training",
    "validation",
    "final_test",
}


def _section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{key} must be a mapping")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{label} must be a positive integer")
    return value


def _finite_positive(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ConfigurationError(f"{label} must be finite and positive")
    return float(value)


def validate_federated_config(config: Mapping[str, Any]) -> None:
    """Reject configurations outside the scientifically fixed FedAvg reference."""

    for key in (
        "name",
        "seed",
        "mode",
        "execution",
        "protocol",
        "device",
        "output_root",
        "pairing_group",
        "dataset",
        "model",
        "training",
        "subset",
        "federated",
        "seed_streams",
        "acceptance",
    ):
        if key not in config:
            raise ConfigurationError(f"missing required setting: {key}")
    if not isinstance(config["name"], str) or not config["name"].strip():
        raise ConfigurationError("name must be a non-empty string")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise ConfigurationError("seed must be an integer")
    if config["mode"] != "scientific_evaluation":
        raise ConfigurationError("federated reference requires mode: scientific_evaluation")
    if config["execution"] != "federated":
        raise ConfigurationError("execution must be federated")
    if config["protocol"] != "independent_evaluation":
        raise ConfigurationError("federated reference requires independent_evaluation")
    if config["device"] not in {"cpu", "cuda"}:
        raise ConfigurationError("device must be cpu or cuda")
    if not isinstance(config["output_root"], str) or not config["output_root"]:
        raise ConfigurationError("output_root must be a path string")
    if not isinstance(config["pairing_group"], str) or not config["pairing_group"]:
        raise ConfigurationError("pairing_group must identify paired experiments")

    dataset = _section(config, "dataset")
    expected_dataset = {
        "name": "shd",
        "train_file": "shd_train.h5",
        "test_file": "shd_test.h5",
        "validation_file": None,
        "classes": 20,
        "raw_channels": 700,
        "input_features": 140,
        "frequency_bin_factor": 5,
        "temporal_bin_ms": 10.0,
        "validation_fraction": 0.1,
    }
    for key, expected in expected_dataset.items():
        if dataset.get(key) != expected:
            raise ConfigurationError(f"dataset.{key} must be {expected!r}")
    if not isinstance(dataset.get("root"), str) or not dataset["root"]:
        raise ConfigurationError("dataset.root must be a path string")

    model = _section(config, "model")
    if model.get("name") != "lif_2layer" or model.get("hidden_dims") != [256, 256]:
        raise ConfigurationError("federated reference requires the 256/256 lif_2layer model")
    if model.get("dropout") != 0.4 or model.get("batch_normalization") is not False or model.get("bias") is not True:
        raise ConfigurationError("federated reference requires dropout 0.4, no batch normalization, and bias")
    attention = _section(model, "attention")
    if attention.get("variant") != "none":
        raise ConfigurationError("DCLS and PfA mechanisms are unsupported in the FedAvg reference")
    neuron = _section(model, "neuron")
    surrogate = _section(neuron, "surrogate")
    expected_neuron = {
        "name": "spikingjelly_lif",
        "tau_ms": 10.05,
        "threshold": 1.0,
        "reset": "subtract",
        "detach_reset": True,
    }
    for key, expected in expected_neuron.items():
        if neuron.get(key) != expected:
            raise ConfigurationError(f"model.neuron.{key} must be {expected!r}")
    if surrogate.get("name") != "atan" or surrogate.get("alpha") != 5.0:
        raise ConfigurationError("federated reference requires ATan surrogate alpha 5")

    subset = _section(config, "subset")
    if any(subset.get(key) != 0 for key in ("train_examples", "validation_examples", "test_examples")):
        raise ConfigurationError("scientific federated evaluation requires complete eligible dataset splits")
    if subset.get("stratified") is not True:
        raise ConfigurationError("SHD validation selection must be stratified")
    training = _section(config, "training")
    batch_limit_keys = ("max_train_batches", "max_validation_batches", "max_test_batches")
    if any(training.get(key) is not None for key in batch_limit_keys):
        raise ConfigurationError("scientific federated evaluation does not permit batch limits")

    federation = _section(config, "federated")
    if federation.get("algorithm") != "fedavg":
        raise ConfigurationError("unsupported federated algorithm")
    clients = _positive_integer(federation.get("clients"), "federated.clients")
    if clients < 2:
        raise ConfigurationError("federated.clients must be at least two")
    rounds = _positive_integer(federation.get("rounds"), "federated.rounds")
    if rounds != 100:
        raise ConfigurationError("scientific federated evaluation requires 100 communication rounds")
    if _positive_integer(federation.get("local_epochs"), "federated.local_epochs") != 1:
        raise ConfigurationError("federated.local_epochs must be one")
    if _positive_integer(federation.get("local_batch_size"), "federated.local_batch_size") != 32:
        raise ConfigurationError("federated.local_batch_size must be 32")
    if federation.get("optimizer") != "adam":
        raise ConfigurationError("federated optimizer must be adam")
    if _finite_positive(federation.get("learning_rate"), "federated.learning_rate") != 0.001:
        raise ConfigurationError("federated.learning_rate must be 0.001")
    if federation.get("weight_decay") != 0.0:
        raise ConfigurationError("federated.weight_decay must be zero")
    if _finite_positive(federation.get("gradient_clip"), "federated.gradient_clip") != 1.0:
        raise ConfigurationError("federated.gradient_clip must be one")
    if federation.get("learning_rate_scheduler") is not None:
        raise ConfigurationError("learning-rate scheduling is unsupported")
    if federation.get("client_sampling") != "without_replacement":
        raise ConfigurationError("client sampling must be without replacement")
    if federation.get("retain_optimizer_state") is not False:
        raise ConfigurationError("client optimizer state must not persist")
    if federation.get("official_test_evaluation_during_rounds") is not False:
        raise ConfigurationError("official-test evaluation during communication rounds is prohibited")

    participation = federation.get("participation_fraction")
    if (
        not isinstance(participation, (int, float))
        or isinstance(participation, bool)
        or not math.isfinite(participation)
        or not 0 < participation <= 1
    ):
        raise ConfigurationError("federated.participation_fraction must be in (0, 1]")
    selected = _positive_integer(federation.get("clients_per_round"), "federated.clients_per_round")
    calculated = clients * float(participation)
    if not calculated.is_integer() or selected != int(calculated):
        raise ConfigurationError("participation must produce the configured integer client count")
    if selected > clients:
        raise ConfigurationError("selected-client count exceeds total clients")
    if float(participation) not in {0.25, 0.5} or selected not in {5, 10}:
        raise ConfigurationError("reference participation must be 0.25 or 0.50")

    partition = _section(federation, "partition")
    if partition.get("method") != "label_dirichlet":
        raise ConfigurationError("partition method must be label_dirichlet")
    if _finite_positive(partition.get("alpha"), "federated.partition.alpha") != 0.5:
        raise ConfigurationError("Dirichlet alpha must be 0.5")
    if _positive_integer(
        partition.get("minimum_examples_per_client"), "federated.partition.minimum_examples_per_client"
    ) != 32:
        raise ConfigurationError("minimum client size must be 32")
    _positive_integer(partition.get("maximum_attempts"), "federated.partition.maximum_attempts")

    if not isinstance(federation.get("data_loader_workers"), int) or federation["data_loader_workers"] < 0:
        raise ConfigurationError("federated.data_loader_workers must be a non-negative integer")
    if not isinstance(federation.get("persistent_workers"), bool):
        raise ConfigurationError("federated.persistent_workers must be boolean")
    if federation["persistent_workers"] and federation["data_loader_workers"] == 0:
        raise ConfigurationError("persistent workers require at least one data-loader worker")

    communication = _section(federation, "communication")
    if communication.get("model_downloads_per_selected_client") != 1:
        raise ConfigurationError("one model download per selected client is required")
    if communication.get("model_uploads_per_selected_client") != 1:
        raise ConfigurationError("one model upload per selected client is required")
    if (
        communication.get("include_optimizer_state") is not False
        or communication.get("include_dataset_transfer") is not False
    ):
        raise ConfigurationError("logical communication excludes optimizer state and dataset transfer")

    streams = _section(config, "seed_streams")
    if set(streams) != REQUIRED_STREAMS:
        raise ConfigurationError(f"seed_streams must contain exactly {sorted(REQUIRED_STREAMS)}")
    if any(not isinstance(value, str) or not value for value in streams.values()):
        raise ConfigurationError("seed stream identities must be non-empty strings")
    if len(set(streams.values())) != len(streams):
        raise ConfigurationError("seed stream identities must be distinct")

    acceptance = _section(config, "acceptance")
    if acceptance.get("expected_model_class") != "AudioLIFSNN":
        raise ConfigurationError("acceptance.expected_model_class must be AudioLIFSNN")
    if acceptance.get("reference_test_accuracy") is not None or acceptance.get("absolute_tolerance") is not None:
        raise ConfigurationError("the FedAvg reference has no verified reproduction target")


def paired_configuration_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields that must match across paired participation treatments."""

    federation = dict(_section(config, "federated"))
    federation.pop("participation_fraction", None)
    federation.pop("clients_per_round", None)
    return {
        "mode": config.get("mode"),
        "execution": config.get("execution"),
        "protocol": config.get("protocol"),
        "pairing_group": config.get("pairing_group"),
        "dataset": dict(_section(config, "dataset")),
        "model": dict(_section(config, "model")),
        "subset": dict(_section(config, "subset")),
        "federated": federation,
        "seed_streams": dict(_section(config, "seed_streams")),
    }
