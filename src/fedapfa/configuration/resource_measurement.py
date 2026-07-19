"""Strict protocol and manifest rules for resource measurement."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path

from .federated_manifest import FEDERATED_SEEDS
from .federated_validation import validate_federated_config
from .loader import load_resolved_config
from .manifest import ManifestTask, _manifest_mapping
from .validation import ConfigurationError

RESOURCE_EXPERIMENTS = (
    ("shd_lif_client_resource", "shd", "shd/lif_client_resource.yaml"),
    ("ssc_lif_128_client_resource", "ssc", "ssc/lif_128_client_resource.yaml"),
)
MEASUREMENT_KEYS = {
    "enabled",
    "sampling_backend",
    "sampling_interval_ms",
    "idle_before_seconds",
    "idle_after_seconds",
    "require_power",
    "require_gpu_utilization",
    "require_memory_utilization",
    "require_cuda_events",
    "maximum_sample_gap_multiplier",
    "timing_reconciliation_tolerance_seconds",
    "boundary_reconciliation_tolerance_joules",
    "calibration",
}
CALIBRATION_KEYS = {
    "paired_repetitions",
    "maximum_median_overhead_fraction",
    "minimum_samples_per_client",
    "minimum_client_fraction",
}
COST_KEYS = {
    "fitting_seeds",
    "evaluation_seed",
    "historical_weight_candidates",
    "ridge_regularization",
    "percentage_denominator_floor",
    "rank_correlation_tolerance",
    "minimum_runtime_error_improvement_fraction",
    "prediction_time_fraction_limit",
}


def _require_equal(value, expected, message: str) -> None:
    if value != expected:
        raise ConfigurationError(message)


def validate_resource_measurement_config(config: Mapping) -> None:
    """Reject every topology or scientific protocol outside the declared collection."""

    validate_federated_config(config)
    dataset = config["dataset"]
    federation = config["federated"]
    model = config["model"]
    parallel = config.get("parallel_execution")
    if dataset.get("name") not in {"shd", "ssc"}:
        raise ConfigurationError("resource measurement dataset must be SHD or SSC")
    if config.get("seed") not in FEDERATED_SEEDS:
        raise ConfigurationError(f"resource measurement seed must be one of {list(FEDERATED_SEEDS)}")
    for key, expected in {
        "mode": "scientific_evaluation",
        "execution": "federated",
        "protocol": "independent_evaluation",
        "device": "cuda",
    }.items():
        _require_equal(config.get(key), expected, f"resource protocol requires {key}={expected!r}")
    if not isinstance(parallel, Mapping):
        raise ConfigurationError("resource measurement requires parallel_execution")
    required_topology = {
        "node_count": 1,
        "device_count": 1,
        "client_processes_per_device": 1,
        "process_count": 1,
        "control_backend": "nccl",
        "cuda_process_service": "none",
        "client_assignment": "selected_order_round_robin",
        "aggregation_order": "selected_client_order",
        "synchronize_each_round": True,
    }
    _require_equal(
        dict(parallel), required_topology, "resource measurement requires the authoritative one-device path"
    )
    common_federated = {
        "algorithm": "fedavg",
        "rounds": 100,
        "clients": 20,
        "clients_per_round": 10,
        "participation_fraction": 0.5,
        "local_epochs": 1,
        "drop_last_local_batch": False,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "aggregation_weighting": "example_count",
        "checkpoint_selection": "best_validation",
        "official_test_evaluation_during_rounds": False,
        "client_sampling": "without_replacement",
        "retain_optimizer_state": False,
        "weight_decay": 0.0,
        "gradient_clip": 1.0,
        "learning_rate_reduction_rounds": [],
        "learning_rate_reduction_factor": 1.0,
        "record_extended_diagnostics": False,
        "data_loader_workers": 8,
        "persistent_workers": True,
        "pin_memory": False,
        "prefetch_factor": 2,
        "non_blocking_transfer": True,
    }
    for key, expected in common_federated.items():
        _require_equal(federation.get(key), expected, f"resource protocol requires federated.{key}={expected!r}")
    partition = federation.get("partition", {})
    _require_equal(
        partition.get("method"),
        "label_dirichlet",
        "resource protocol requires label-Dirichlet partitioning",
    )
    _require_equal(partition.get("alpha"), 0.5, "resource protocol requires Dirichlet alpha 0.5")
    _require_equal(
        partition.get("minimum_examples_per_client"),
        32,
        "resource protocol requires at least 32 examples per client",
    )
    _require_equal(
        partition.get("maximum_attempts"),
        1000,
        "resource protocol requires the established partition attempt limit",
    )
    for section in (config.get("training", {}),):
        for key in ("max_train_batches", "max_validation_batches", "max_test_batches"):
            if section.get(key) is not None:
                raise ConfigurationError("resource protocol forbids batch caps")
    subset = config.get("subset", {})
    if any(subset.get(key) != 0 for key in ("train_examples", "validation_examples", "test_examples")):
        raise ConfigurationError("resource protocol requires complete dataset collections")
    if dataset["name"] == "shd":
        _require_equal(dataset.get("train_file"), "shd_train.h5", "SHD training file is incompatible")
        _require_equal(dataset.get("test_file"), "shd_test.h5", "SHD test file is incompatible")
        _require_equal(dataset.get("classes"), 20, "SHD resource protocol requires 20 classes")
        _require_equal(federation.get("local_batch_size"), 32, "SHD resource protocol requires batch size 32")
        _require_equal(model.get("name"), "lif_2layer", "SHD resource protocol requires the established LIF model")
        _require_equal(model.get("hidden_dims"), [256, 256], "SHD resource protocol requires 256/256 hidden units")
        _require_equal(
            dataset.get("validation_fraction"), 0.1, "SHD resource protocol requires stratified validation"
        )
        _require_equal(
            dataset.get("validation_file"),
            None,
            "SHD resource protocol derives validation from training data",
        )
    else:
        _require_equal(dataset.get("train_file"), "ssc_train.h5", "SSC training file is incompatible")
        _require_equal(dataset.get("test_file"), "ssc_test.h5", "SSC test file is incompatible")
        _require_equal(dataset.get("classes"), 35, "SSC resource protocol requires 35 classes")
        _require_equal(federation.get("local_batch_size"), 256, "SSC resource protocol requires batch size 256")
        _require_equal(
            model.get("name"),
            "lif_2layer_128",
            "SSC resource protocol requires the established LIF-128 model",
        )
        _require_equal(
            model.get("hidden_dims"), [128, 128], "SSC resource protocol requires 128/128 hidden units"
        )
        _require_equal(
            dataset.get("validation_file"),
            "ssc_valid.h5",
            "SSC resource protocol requires official validation",
        )
        _require_equal(
            dataset.get("validation_fraction"),
            0.0,
            "SSC resource protocol must not derive internal validation",
        )
    for key, expected in {
        "raw_channels": 700,
        "input_features": 140,
        "frequency_bin_factor": 5,
        "temporal_bin_ms": 10.0,
    }.items():
        _require_equal(dataset.get(key), expected, f"resource protocol requires dataset.{key}={expected!r}")
    model_common = {
        "dropout": 0.4,
        "batch_normalization": False,
        "bias": True,
    }
    for key, expected in model_common.items():
        _require_equal(model.get(key), expected, f"resource protocol requires model.{key}={expected!r}")
    _require_equal(model.get("attention"), {"variant": "none", "lambda": 0.01}, "attention is incompatible")
    _require_equal(
        model.get("neuron"),
        {
            "name": "spikingjelly_lif",
            "tau_ms": 10.05,
            "threshold": 1.0,
            "reset": "subtract",
            "detach_reset": True,
            "surrogate": {"name": "atan", "alpha": 5.0},
        },
        "LIF neuron configuration is incompatible",
    )
    execution_measurement = config.get("execution_measurement", {})
    if execution_measurement.get("profiler_enabled") or execution_measurement.get(
        "record_device_utilization"
    ):
        raise ConfigurationError("resource protocol requires only the declared measurement backend")
    measurement = config.get("resource_measurement")
    if not isinstance(measurement, Mapping) or set(measurement) != MEASUREMENT_KEYS:
        raise ConfigurationError(f"resource_measurement must contain exactly {sorted(MEASUREMENT_KEYS)}")
    expected_measurement = {
        "enabled": True,
        "sampling_backend": "nvml",
        "sampling_interval_ms": 100,
        "idle_before_seconds": 30,
        "idle_after_seconds": 30,
        "require_power": True,
        "require_gpu_utilization": True,
        "require_memory_utilization": True,
        "require_cuda_events": True,
        "maximum_sample_gap_multiplier": 2.5,
        "timing_reconciliation_tolerance_seconds": 0.000002,
        "boundary_reconciliation_tolerance_joules": 0.000001,
    }
    for key, expected in expected_measurement.items():
        _require_equal(measurement.get(key), expected, f"resource measurement requires {key}={expected!r}")
    calibration = measurement.get("calibration")
    if not isinstance(calibration, Mapping) or set(calibration) != CALIBRATION_KEYS:
        raise ConfigurationError(f"resource calibration must contain exactly {sorted(CALIBRATION_KEYS)}")
    _require_equal(calibration.get("paired_repetitions"), 10, "resource calibration requires ten pairs")
    _require_equal(calibration.get("maximum_median_overhead_fraction"), 0.02, "calibration overhead limit must be 2%")
    _require_equal(calibration.get("minimum_samples_per_client"), 10, "calibration requires ten samples per client")
    _require_equal(calibration.get("minimum_client_fraction"), 0.9, "calibration client coverage must be 90%")
    cost = config.get("cost_estimation")
    if not isinstance(cost, Mapping) or set(cost) != COST_KEYS:
        raise ConfigurationError(f"cost_estimation must contain exactly {sorted(COST_KEYS)}")
    _require_equal(cost.get("fitting_seeds"), [7, 17], "cost fitting seeds must be 7 and 17")
    _require_equal(cost.get("evaluation_seed"), 27, "cost evaluation seed must be 27")
    historical_weights = cost.get("historical_weight_candidates")
    if (
        not isinstance(historical_weights, list)
        or len(historical_weights) < 2
        or any(not 0 < float(value) < 1 for value in historical_weights)
        or [float(value) for value in historical_weights]
        != sorted({float(value) for value in historical_weights})
    ):
        raise ConfigurationError(
            "historical weighting candidates must be unique increasing values between zero and one"
        )
    penalties = cost.get("ridge_regularization")
    if not isinstance(penalties, list) or not penalties or any(float(value) <= 0 for value in penalties):
        raise ConfigurationError("ridge regularization candidates must be positive")
    _require_equal(
        cost.get("minimum_runtime_error_improvement_fraction"),
        0.05,
        "spike-history runtime improvement threshold must be 5%",
    )
    _require_equal(
        cost.get("rank_correlation_tolerance"),
        0.01,
        "spike-history rank tolerance is incompatible",
    )
    _require_equal(
        cost.get("prediction_time_fraction_limit"),
        0.001,
        "prediction-time threshold is incompatible",
    )
    _require_equal(
        cost.get("percentage_denominator_floor"),
        0.001,
        "percentage denominator floor is incompatible",
    )
    if config.get("metadata", {}).get("purpose") != "resource_measurement":
        raise ConfigurationError("resource configuration purpose is incompatible")
    if config.get("metadata", {}).get("experiment") != config.get("name"):
        raise ConfigurationError("resource metadata experiment must match name")
    root = str(config.get("output_root", "")).replace("\\", "/").rstrip("/")
    if root != "runs/resource_measurement" and not root.endswith("/runs/resource_measurement"):
        raise ConfigurationError("resource output root must isolate the collection")


def load_resource_measurement_config(path: str | Path) -> dict:
    config = load_resolved_config(path)
    validate_resource_measurement_config(config)
    return config


def load_resource_measurement_manifest(path: str | Path) -> list[ManifestTask]:
    manifest_path = Path(path).resolve()
    manifest = _manifest_mapping(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("collection") != "resource_measurement":
        raise ValueError("resource manifest identity is incompatible")
    seeds = _manifest_mapping((manifest_path.parent / manifest.get("seeds_file", "")).resolve()).get("seeds")
    if seeds != list(FEDERATED_SEEDS):
        raise ValueError(f"resource seeds must be exactly {list(FEDERATED_SEEDS)}")
    entries = manifest.get("experiments")
    if not isinstance(entries, list) or len(entries) != len(RESOURCE_EXPERIMENTS):
        raise ValueError("resource manifest must contain exactly two experiments")
    tasks = []
    identities = set()
    for entry, (experiment, dataset, relative_path) in zip(entries, RESOURCE_EXPERIMENTS, strict=True):
        if entry != {"id": experiment, "config": relative_path, "mandatory": True}:
            raise ValueError("resource manifest experiment order or identity is incompatible")
        config_path = (manifest_path.parent / relative_path).resolve()
        if manifest_path.parent not in config_path.parents:
            raise ValueError("resource config path escapes its collection")
        template = load_resource_measurement_config(config_path)
        if template["name"] != experiment or template["dataset"]["name"] != dataset:
            raise ValueError("resource workload identity is incompatible")
        for seed in seeds:
            config = copy.deepcopy(template)
            config["seed"] = seed
            validate_resource_measurement_config(config)
            identity = (experiment, seed)
            if identity in identities:
                raise ValueError("resource output identity is duplicated")
            identities.add(identity)
            tasks.append(
                ManifestTask(experiment, config_path, seed, dataset, config["mode"], config["protocol"], config)
            )
    if len(tasks) != 6:
        raise ValueError("resource manifest must expand to exactly six tasks")
    return tasks
