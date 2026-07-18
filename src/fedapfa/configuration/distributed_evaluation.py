"""Strict configuration rules for single-node synchronous distributed FedAvg."""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .federated_manifest import FEDERATED_SEEDS
from .federated_validation import validate_federated_config
from .loader import load_resolved_config
from .manifest import ManifestTask, _manifest_mapping
from .validation import ConfigurationError

DISTRIBUTED_DEVICE_COUNTS = (1, 2, 4)
DISTRIBUTED_EXPERIMENTS = (
    ("shd_lif_fedavg_1_gpu", "shd", 1),
    ("shd_lif_fedavg_2_gpu", "shd", 2),
    ("shd_lif_fedavg_4_gpu", "shd", 4),
    ("ssc_lif_128_fedavg_1_gpu", "ssc", 1),
    ("ssc_lif_128_fedavg_2_gpu", "ssc", 2),
    ("ssc_lif_128_fedavg_4_gpu", "ssc", 4),
    ("cifar10_svgg9_bntt_noniid_1_gpu", "cifar10", 1),
    ("cifar10_svgg9_bntt_noniid_2_gpu", "cifar10", 2),
)
DEVICE_CAPACITY_EXPERIMENTS = (
    ("shd_lif_fedavg_1_gpu_1_client_process", 1),
    ("shd_lif_fedavg_1_gpu_2_client_processes", 2),
    ("shd_lif_fedavg_1_gpu_4_client_processes", 4),
)
PARALLEL_KEYS = {
    "node_count",
    "device_count",
    "client_processes_per_device",
    "process_count",
    "control_backend",
    "cuda_process_service",
    "client_assignment",
    "aggregation_order",
    "synchronize_each_round",
}
MEASUREMENT_KEYS = {
    "profiler_enabled",
    "profiled_rounds",
    "record_cuda_memory",
    "record_device_utilization",
    "utilization_interval_seconds",
}


def _parallel(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("parallel_execution")
    if not isinstance(value, Mapping):
        raise ConfigurationError("parallel_execution must be a mapping")
    return value


def validate_parallel_execution(config: Mapping[str, Any]) -> None:
    """Validate exclusive-device and same-device MPS process topology."""

    parallel = _parallel(config)
    if set(parallel) != PARALLEL_KEYS:
        raise ConfigurationError(f"parallel_execution must contain exactly {sorted(PARALLEL_KEYS)}")
    if parallel.get("node_count") != 1:
        raise ConfigurationError("distributed scientific execution requires node_count 1")
    devices = parallel.get("device_count")
    per_device = parallel.get("client_processes_per_device")
    processes = parallel.get("process_count")
    if devices not in DISTRIBUTED_DEVICE_COUNTS:
        raise ConfigurationError("parallel_execution.device_count must be 1, 2, or 4")
    if per_device not in {1, 2, 4}:
        raise ConfigurationError("parallel_execution.client_processes_per_device must be 1, 2, or 4")
    if processes != devices * per_device:
        raise ConfigurationError(
            "parallel_execution.process_count must equal device_count times client_processes_per_device"
        )
    selected = config["federated"]["clients_per_round"]
    if devices > selected:
        raise ConfigurationError("physical device count cannot exceed selected clients")
    if processes > selected:
        raise ConfigurationError("client process count cannot exceed selected clients")
    backend = parallel.get("control_backend")
    service = parallel.get("cuda_process_service")
    if per_device == 1:
        if backend != "nccl" or service != "none":
            raise ConfigurationError("exclusive physical-device execution requires NCCL and CUDA process service none")
    elif backend != "gloo" or service != "mps":
        raise ConfigurationError("same-device client packing requires Gloo coordination and CUDA MPS")
    if parallel.get("client_assignment") != "selected_order_round_robin":
        raise ConfigurationError("unsupported distributed client assignment")
    if parallel.get("aggregation_order") != "selected_client_order":
        raise ConfigurationError("unsupported distributed aggregation order")
    if parallel.get("synchronize_each_round") is not True:
        raise ConfigurationError("distributed FedAvg requires synchronous communication rounds")

    measurement = config.get("execution_measurement")
    if not isinstance(measurement, Mapping) or set(measurement) != MEASUREMENT_KEYS:
        raise ConfigurationError(f"execution_measurement must contain exactly {sorted(MEASUREMENT_KEYS)}")
    if not isinstance(measurement.get("profiler_enabled"), bool):
        raise ConfigurationError("execution_measurement.profiler_enabled must be boolean")
    profiled = measurement.get("profiled_rounds")
    rounds = config["federated"]["rounds"]
    if not isinstance(profiled, list) or any(
        not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= rounds for value in profiled
    ):
        raise ConfigurationError("execution_measurement.profiled_rounds must contain valid communication rounds")
    if profiled != sorted(set(profiled)):
        raise ConfigurationError("execution_measurement.profiled_rounds must be sorted and unique")
    if measurement["profiler_enabled"] != bool(profiled):
        raise ConfigurationError("profiling requires explicit profiled rounds and disabled profiling requires none")
    if measurement.get("record_cuda_memory") is not True:
        raise ConfigurationError("distributed execution must record CUDA memory")
    if measurement.get("record_device_utilization") is not True:
        raise ConfigurationError("distributed execution must record physical-device utilization")
    interval = measurement.get("utilization_interval_seconds")
    if interval != 2:
        raise ConfigurationError("distributed physical-device utilization interval must be 2 seconds")


def _validate_workload(config: Mapping[str, Any]) -> None:
    dataset = config["dataset"]["name"]
    federation = config["federated"]
    if dataset == "shd":
        expected = {
            "clients": 20,
            "clients_per_round": 10,
            "local_epochs": 1,
            "local_batch_size": 32,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "aggregation_weighting": "example_count",
            "checkpoint_selection": "best_validation",
        }
    elif dataset == "ssc":
        expected = {
            "clients": 20,
            "clients_per_round": 10,
            "local_epochs": 1,
            "local_batch_size": 256,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "aggregation_weighting": "example_count",
            "checkpoint_selection": "best_validation",
        }
        if config["dataset"].get("validation_file") != "ssc_valid.h5":
            raise ConfigurationError("SSC distributed evaluation requires the official validation collection")
    elif dataset == "cifar10":
        expected = {
            "clients": 10,
            "clients_per_round": 2,
            "local_epochs": 5,
            "local_batch_size": 32,
            "optimizer": "sgd",
            "learning_rate": 0.1,
            "weight_decay": 0.0001,
            "gradient_clip": None,
            "aggregation_weighting": "uniform",
            "checkpoint_selection": "final_round",
        }
        partition = federation["partition"]
        if partition.get("method") != "fedsnn_balanced_label_dirichlet" or partition.get("alpha") != 0.5:
            raise ConfigurationError("CIFAR-10 distributed evaluation requires balanced label-Dirichlet alpha 0.5")
    else:
        raise ConfigurationError(f"unsupported distributed workload dataset: {dataset}")
    for key, expected_value in expected.items():
        if federation.get(key) != expected_value:
            raise ConfigurationError(
                f"distributed {dataset.upper()} workload requires federated.{key}={expected_value!r}"
            )
    if federation.get("rounds") != 100:
        raise ConfigurationError("distributed workloads require 100 communication rounds")
    if dataset != "cifar10" and federation.get("participation_fraction") != 0.5:
        raise ConfigurationError("distributed event-audio workloads require 50% participation")
    if dataset == "cifar10" and federation.get("participation_fraction") != 0.2:
        raise ConfigurationError("distributed CIFAR-10 workload requires 20% participation")


def validate_distributed_evaluation_config(config: Mapping[str, Any]) -> None:
    """Require a validated workload and explicit single-node execution placement."""

    validate_federated_config(config)
    validate_parallel_execution(config)
    _validate_workload(config)
    if config.get("metadata", {}).get("experiment") != config.get("name"):
        raise ConfigurationError("distributed metadata experiment must match the execution identity")
    normalized_root = str(config.get("output_root", "")).replace("\\", "/").rstrip("/")
    allowed_roots = ("runs/distributed_evaluation", "runs/device_capacity_evaluation")
    if not any(normalized_root == value or normalized_root.endswith(f"/{value}") for value in allowed_roots):
        raise ConfigurationError("distributed output_root must isolate distributed or device-capacity evaluation")


def load_distributed_evaluation_config(path: str | Path) -> dict:
    config = load_resolved_config(path)
    validate_distributed_evaluation_config(config)
    return config


def distributed_scientific_identity(config: Mapping[str, Any]) -> dict:
    """Return scientific fields that must not change with execution placement."""

    identity = copy.deepcopy(dict(config))
    for key in (
        "name",
        "output_root",
        "resume",
        "metadata",
        "pairing_group",
        "parallel_execution",
        "execution_measurement",
    ):
        identity.pop(key, None)
    return identity


def distributed_execution_identity(config: Mapping[str, Any]) -> dict:
    """Return the configured resource and process topology identity."""

    return {
        "parallel_execution": copy.deepcopy(dict(_parallel(config))),
        "process_to_device_mapping": process_device_mapping(
            int(config["parallel_execution"]["device_count"]),
            int(config["parallel_execution"]["process_count"]),
        ),
        "execution_measurement": copy.deepcopy(dict(config["execution_measurement"])),
    }


def process_device_mapping(device_count: int, process_count: int) -> list[dict[str, int]]:
    """Resolve deterministic process-to-device and per-device slot positions."""

    if device_count not in DISTRIBUTED_DEVICE_COUNTS or process_count <= 0 or process_count % device_count:
        raise ValueError("process-to-device mapping requires a supported divisible topology")
    return [
        {
            "process_rank": rank,
            "device_index": rank % device_count,
            "device_slot": rank // device_count,
        }
        for rank in range(process_count)
    ]


def _load_manifest(path: str | Path, collection: str, expected: tuple[tuple, ...]) -> list[ManifestTask]:
    manifest_path = Path(path).resolve()
    manifest = _manifest_mapping(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("collection") != collection:
        raise ValueError(f"manifest must declare schema_version: 1 and collection: {collection}")
    seeds_name = manifest.get("seeds_file")
    if not isinstance(seeds_name, str) or not seeds_name:
        raise ValueError("distributed manifest requires seeds_file")
    seeds = _manifest_mapping((manifest_path.parent / seeds_name).resolve()).get("seeds")
    if seeds != list(FEDERATED_SEEDS):
        raise ValueError(f"distributed evaluation seeds must be exactly {list(FEDERATED_SEEDS)}")
    entries = manifest.get("experiments")
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise ValueError(f"{collection} manifest must contain exactly {len(expected)} treatments")
    tasks: list[ManifestTask] = []
    templates: dict[str, list[dict]] = {}
    for specification, entry in zip(expected, entries, strict=True):
        name, dataset, device_count = specification
        if not isinstance(entry, Mapping) or entry.get("mandatory") is not True or entry.get("id") != name:
            raise ValueError("distributed manifest experiment order or identity is incompatible")
        config_name = entry.get("config")
        if not isinstance(config_name, str) or not config_name:
            raise ValueError(f"distributed experiment {name} requires a config path")
        config_path = (manifest_path.parent / config_name).resolve()
        if manifest_path.parent not in config_path.parents:
            raise ValueError("distributed manifest config path escapes its collection")
        template = load_distributed_evaluation_config(config_path)
        if template["name"] != name or template["dataset"]["name"] != dataset:
            raise ValueError("distributed manifest workload identity is incompatible")
        if template["parallel_execution"]["device_count"] != device_count:
            raise ValueError("distributed manifest device count is incompatible")
        templates.setdefault(dataset, []).append(template)
        for seed in seeds:
            config = copy.deepcopy(template)
            config["seed"] = seed
            validate_distributed_evaluation_config(config)
            tasks.append(
                ManifestTask(name, config_path, seed, dataset, config["mode"], config["protocol"], config)
            )
    for dataset, values in templates.items():
        if len({repr(distributed_scientific_identity(value)) for value in values}) != 1:
            raise ValueError(f"{dataset} treatments differ outside execution-placement fields")
    return tasks


def load_distributed_evaluation_manifest(path: str | Path) -> list[ManifestTask]:
    tasks = _load_manifest(path, "distributed_evaluation", DISTRIBUTED_EXPERIMENTS)
    counts = Counter(task.dataset for task in tasks)
    if len(tasks) != 24 or counts != {"shd": 9, "ssc": 9, "cifar10": 6}:
        raise ValueError("distributed evaluation manifest must expand to 9 SHD, 9 SSC, and 6 CIFAR-10 tasks")
    return tasks


def load_device_capacity_manifest(path: str | Path) -> list[ManifestTask]:
    expected = tuple((name, "shd", 1) for name, _ in DEVICE_CAPACITY_EXPERIMENTS)
    tasks = _load_manifest(path, "device_capacity_evaluation", expected)
    expected_processes = [value for _, value in DEVICE_CAPACITY_EXPERIMENTS for _ in FEDERATED_SEEDS]
    if [task.config["parallel_execution"]["process_count"] for task in tasks] != expected_processes:
        raise ValueError("device-capacity manifest process counts are incompatible")
    return tasks
