"""Strict scientific configuration for scheduling and hierarchical reduction."""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from fedapfa.scheduling.base import (
    ASSIGNMENT_TIE_BREAKING_VERSION,
    EVENT_STRUCTURE_FEATURES,
    SCHEDULING_STRATEGIES,
)
from fedapfa.scheduling.runtime_cost_model import (
    MODEL_CONFIGURATION_KEYS,
    FrozenEventStructureModel,
)

from .distributed_evaluation import MEASUREMENT_KEYS
from .experiment_id import experiment_id
from .federated_validation import validate_federated_config
from .loader import load_resolved_config
from .manifest import ManifestTask, _manifest_mapping
from .validation import ConfigurationError

EVALUATION_SEEDS = (37, 47, 57)
AGGREGATION_TOPOLOGIES = ("flat_ordered", "node_hierarchical")
EVALUATION_COLLECTIONS = ("scheduling_evaluation", "hierarchical_reduction_evaluation")
PARALLEL_KEYS = {
    "node_count",
    "devices_per_node",
    "device_count",
    "client_processes_per_device",
    "process_count",
    "control_backend",
    "cuda_process_service",
    "client_assignment",
    "aggregation_order",
    "aggregation_topology",
    "rank_mapping",
    "synchronize_each_round",
}
SCHEDULER_KEYS = {
    "strategy",
    "cost_model",
    "tie_breaking_version",
    "predictions_before_client_training",
    "training_data_only",
}
AGGREGATION_KEYS = {
    "topology",
    "order_policy",
    "accumulator_dtype",
    "absolute_tolerance",
    "relative_tolerance",
    "material_runtime_regression_fraction",
}
EVALUATION_KEYS = {
    "collection",
    "comparison_reference",
    "within_allocation_execution",
    "independent_process_invocation",
}
RESOLVED_PAIR_WHITELIST_VERSION = "resolved_leaf_paths_v1"
RESOLVED_PAIR_ALLOWED_DIFFERENCES = {
    "scheduling_evaluation": {
        "name",
        "metadata.experiment",
        "output_root",
        "scheduler.strategy",
        "parallel_execution.client_assignment",
    },
    "hierarchical_reduction_evaluation": {
        "name",
        "metadata.experiment",
        "output_root",
        "aggregation_execution.topology",
        "parallel_execution.aggregation_topology",
    },
}
RESOLVED_PAIRED_INVARIANT_PATHS = {
    "dataset_and_preprocessing": ("dataset",),
    "dataset_split": ("subset", "dataset.validation_fraction", "seed_streams.split"),
    "client_partition": ("federated.partition", "seed_streams.partition"),
    "client_population": ("federated.clients",),
    "participation": ("federated.participation_fraction", "federated.client_sampling"),
    "selected_client_count": ("federated.clients_per_round",),
    "communication_rounds": ("federated.rounds",),
    "model_architecture": ("model",),
    "model_initialization": ("seed", "seed_streams.model_initialization"),
    "optimizer": (
        "federated.optimizer",
        "federated.weight_decay",
        "federated.gradient_clip",
        "federated.retain_optimizer_state",
    ),
    "learning_rate_policy": (
        "federated.learning_rate",
        "federated.learning_rate_reduction_rounds",
        "federated.learning_rate_reduction_factor",
    ),
    "local_epochs": ("federated.local_epochs",),
    "local_batch_size": ("federated.local_batch_size",),
    "drop_last_behavior": ("federated.drop_last_local_batch",),
    "client_seed_derivation": ("seed", "seed_streams.client_training"),
    "fedavg_weighting": ("federated.algorithm", "federated.aggregation_weighting"),
    "validation_protocol": (
        "training.max_validation_batches",
        "seed_streams.validation",
        "dataset.validation_file",
    ),
    "checkpoint_selection_protocol": ("federated.checkpoint_selection",),
    "official_test_protocol": (
        "training.max_test_batches",
        "seed_streams.final_test",
        "dataset.test_file",
        "federated.official_test_evaluation_during_rounds",
    ),
    "official_test_access_count": ("federated.official_test_evaluation_during_rounds",),
    "physical_gpu_count": ("parallel_execution.device_count",),
    "process_count": ("parallel_execution.process_count",),
    "processes_per_gpu": ("parallel_execution.client_processes_per_device",),
}

SCHEDULING_EXPERIMENTS = (
    ("shd_lif_fedavg_round_robin", "shd", "round_robin", "flat_ordered"),
    (
        "shd_lif_fedavg_example_count_longest_processing_time",
        "shd",
        "example_count_longest_processing_time",
        "flat_ordered",
    ),
    (
        "shd_lif_fedavg_event_structure_longest_processing_time",
        "shd",
        "event_structure_longest_processing_time",
        "flat_ordered",
    ),
    ("ssc_lif_128_fedavg_round_robin", "ssc", "round_robin", "flat_ordered"),
    (
        "ssc_lif_128_fedavg_example_count_longest_processing_time",
        "ssc",
        "example_count_longest_processing_time",
        "flat_ordered",
    ),
    (
        "ssc_lif_128_fedavg_event_structure_longest_processing_time",
        "ssc",
        "event_structure_longest_processing_time",
        "flat_ordered",
    ),
)
HIERARCHICAL_EXPERIMENTS = (
    (
        "shd_lif_fedavg_flat_ordered",
        "shd",
        "event_structure_longest_processing_time",
        "flat_ordered",
    ),
    (
        "shd_lif_fedavg_node_hierarchical",
        "shd",
        "event_structure_longest_processing_time",
        "node_hierarchical",
    ),
    (
        "ssc_lif_128_fedavg_flat_ordered",
        "ssc",
        "event_structure_longest_processing_time",
        "flat_ordered",
    ),
    (
        "ssc_lif_128_fedavg_node_hierarchical",
        "ssc",
        "event_structure_longest_processing_time",
        "node_hierarchical",
    ),
)


@dataclass(frozen=True)
class EvaluationAllocation:
    collection: str
    dataset: str
    seed: int
    execution_order: tuple[str, ...]
    tasks: tuple[ManifestTask, ...]


def _repository_root(path: Path) -> Path:
    for value in (path.resolve(), *path.resolve().parents):
        if (value / ".git").exists() and (value / "results").is_dir():
            return value
    raise ValueError(f"repository root could not be resolved from {path}")


def _run_directory(config: Mapping) -> Path:
    return Path(config["output_root"]) / experiment_id(dict(config))


def _mapping(config: Mapping, name: str, keys: set[str]) -> Mapping:
    value = config.get(name)
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ConfigurationError(f"{name} must contain exactly {sorted(keys)}")
    return value


def _finite_nonnegative(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ConfigurationError(f"{label} must be finite and nonnegative")
    return float(value)


def evaluation_scientific_identity(config: Mapping) -> dict:
    """Return fields that must be identical between execution treatments."""

    identity = copy.deepcopy(dict(config))
    for key in (
        "name",
        "output_root",
        "resume",
        "metadata",
        "pairing_group",
        "parallel_execution",
        "execution_measurement",
        "scheduler",
        "aggregation_execution",
        "evaluation",
    ):
        identity.pop(key, None)
    return identity


def _resolved_leaf_differences(first, second, path: str = "") -> dict[str, tuple[object, object]]:
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        differences = {}
        for key in sorted(set(first) | set(second)):
            current = f"{path}.{key}" if path else str(key)
            if key not in first or key not in second:
                differences[current] = (first.get(key), second.get(key))
            else:
                differences.update(_resolved_leaf_differences(first[key], second[key], current))
        return differences
    if first != second:
        return {path: (copy.deepcopy(first), copy.deepcopy(second))}
    return {}


def _resolved_value(config: Mapping, path: str):
    value = config
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ConfigurationError(f"resolved paired invariant path is missing: {path}")
        value = value[part]
    return copy.deepcopy(value)


def resolved_paired_invariants(config: Mapping) -> dict:
    """Return named resolved values that must be equal across paired treatments."""

    return {
        name: {path: _resolved_value(config, path) for path in paths}
        for name, paths in RESOLVED_PAIRED_INVARIANT_PATHS.items()
    }


def validate_resolved_evaluation_pair(first: Mapping, second: Mapping, collection: str) -> dict:
    """Allow only declared treatment/run identity differences after YAML resolution."""

    if collection not in RESOLVED_PAIR_ALLOWED_DIFFERENCES:
        raise ConfigurationError("resolved comparison collection is unsupported")
    if first.get("evaluation", {}).get("collection") != collection or second.get("evaluation", {}).get(
        "collection"
    ) != collection:
        raise ConfigurationError("resolved comparison crosses evaluation collections")
    differences = _resolved_leaf_differences(first, second)
    unexpected = sorted(set(differences) - RESOLVED_PAIR_ALLOWED_DIFFERENCES[collection])
    if unexpected:
        raise ConfigurationError(f"resolved paired configurations differ outside the whitelist: {unexpected}")
    first_invariants = resolved_paired_invariants(first)
    second_invariants = resolved_paired_invariants(second)
    if first_invariants != second_invariants:
        raise ConfigurationError("resolved paired scientific or execution invariants differ")
    return {
        "whitelist_version": RESOLVED_PAIR_WHITELIST_VERSION,
        "collection": collection,
        "allowed_difference_paths": sorted(RESOLVED_PAIR_ALLOWED_DIFFERENCES[collection]),
        "observed_difference_paths": sorted(differences),
        "invariants_equal": {name: True for name in RESOLVED_PAIRED_INVARIANT_PATHS},
        "invariant_values": first_invariants,
    }


def evaluation_execution_identity(config: Mapping) -> dict:
    """Return stable placement, scheduler, model, and aggregation resume identity."""

    parallel = dict(config["parallel_execution"])
    return {
        "parallel_execution": copy.deepcopy(parallel),
        "process_to_device_mapping": stable_rank_mapping(
            int(parallel["node_count"]), int(parallel["devices_per_node"])
        ),
        "scheduler": copy.deepcopy(dict(config["scheduler"])),
        "aggregation_execution": copy.deepcopy(dict(config["aggregation_execution"])),
        "execution_measurement": copy.deepcopy(dict(config["execution_measurement"])),
    }


def stable_rank_mapping(node_count: int, devices_per_node: int) -> list[dict[str, int]]:
    if node_count <= 0 or devices_per_node <= 0:
        raise ValueError("rank mapping dimensions must be positive")
    return [
        {
            "global_rank": node_rank * devices_per_node + local_rank,
            "node_rank": node_rank,
            "local_rank": local_rank,
            "local_device_index": local_rank,
        }
        for node_rank in range(node_count)
        for local_rank in range(devices_per_node)
    ]


def validate_evaluation_config(
    config: Mapping,
    *,
    repository_root: str | Path | None = None,
) -> None:
    """Reject topology, predictor, resumption, and scientific-protocol drift."""

    validate_federated_config(config)
    evaluation = _mapping(config, "evaluation", EVALUATION_KEYS)
    collection = evaluation["collection"]
    if collection not in EVALUATION_COLLECTIONS:
        raise ConfigurationError("evaluation collection is unsupported")
    if not isinstance(evaluation["comparison_reference"], str) or not evaluation["comparison_reference"]:
        raise ConfigurationError("evaluation comparison reference is missing")
    if (
        evaluation["within_allocation_execution"] != "counterbalanced_sequential"
        or evaluation["independent_process_invocation"] is not True
    ):
        raise ConfigurationError("evaluation treatments require independent counterbalanced invocations")

    scheduler = _mapping(config, "scheduler", SCHEDULER_KEYS)
    strategy = scheduler["strategy"]
    if strategy not in SCHEDULING_STRATEGIES:
        raise ConfigurationError("unsupported scheduler name")
    if scheduler["tie_breaking_version"] != ASSIGNMENT_TIE_BREAKING_VERSION:
        raise ConfigurationError("scheduler tie-breaking version is incompatible")
    if scheduler["predictions_before_client_training"] is not True or scheduler["training_data_only"] is not True:
        raise ConfigurationError("scheduler predictions must use training data before client execution")
    model_config = scheduler["cost_model"]
    if not isinstance(model_config, Mapping) or set(model_config) != MODEL_CONFIGURATION_KEYS:
        raise ConfigurationError("scheduler cost model is missing or malformed")
    if model_config.get("feature_order") != list(EVENT_STRUCTURE_FEATURES):
        raise ConfigurationError("scheduler predictor feature list is incompatible")
    try:
        FrozenEventStructureModel.load(
            dict(model_config),
            dataset_name=config["dataset"]["name"],
            model_name=config["model"]["name"],
            repository_root=repository_root,
        )
    except (ValueError, OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(str(error)) from error

    aggregation = _mapping(config, "aggregation_execution", AGGREGATION_KEYS)
    topology = aggregation["topology"]
    if topology not in AGGREGATION_TOPOLOGIES:
        raise ConfigurationError("unsupported aggregation topology")
    if aggregation["order_policy"] != "selected_client_order":
        raise ConfigurationError("aggregation order policy must restore selected-client order")
    if aggregation["accumulator_dtype"] != "float64":
        raise ConfigurationError("aggregation accumulator precision must be float64")
    _finite_nonnegative(aggregation["absolute_tolerance"], "aggregation absolute tolerance")
    _finite_nonnegative(aggregation["relative_tolerance"], "aggregation relative tolerance")
    regression = _finite_nonnegative(
        aggregation["material_runtime_regression_fraction"],
        "material runtime regression fraction",
    )
    if regression > 1:
        raise ConfigurationError("material runtime regression fraction cannot exceed one")

    parallel = _mapping(config, "parallel_execution", PARALLEL_KEYS)
    nodes = parallel["node_count"]
    per_node = parallel["devices_per_node"]
    devices = parallel["device_count"]
    processes = parallel["process_count"]
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in (nodes, per_node, devices, processes)
    ):
        raise ConfigurationError("parallel topology dimensions must be positive integers")
    if devices != nodes * per_node or processes != devices:
        raise ConfigurationError("process count is inconsistent with devices and nodes")
    if devices != 4 or processes != 4:
        raise ConfigurationError("evaluation requires exactly four GPUs and four processes")
    if parallel["client_processes_per_device"] != 1:
        raise ConfigurationError("evaluation requires one process per GPU")
    if parallel["control_backend"] != "nccl" or parallel["cuda_process_service"] != "none":
        raise ConfigurationError("evaluation requires NCCL without CUDA MPS")
    if parallel["client_assignment"] != strategy:
        raise ConfigurationError("parallel client assignment differs from scheduler strategy")
    if (
        parallel["aggregation_order"] != "selected_client_order"
        or parallel["aggregation_topology"] != topology
        or parallel["rank_mapping"] != "node_major_local_rank"
        or parallel["synchronize_each_round"] is not True
    ):
        raise ConfigurationError("parallel aggregation or rank mapping is incompatible")

    if collection == "scheduling_evaluation":
        if (nodes, per_node, topology) != (1, 4, "flat_ordered"):
            raise ConfigurationError("scheduling evaluation requires one node with four GPUs")
        expected_root = "runs/scheduling_evaluation"
    else:
        if strategy != "event_structure_longest_processing_time":
            raise ConfigurationError("hierarchical evaluation must hold event-structure scheduling constant")
        if (nodes, per_node) != (2, 2):
            raise ConfigurationError("hierarchical topology must be exactly two nodes by two GPUs")
        expected_root = "runs/hierarchical_reduction_evaluation"
    normalized_root = str(config["output_root"]).replace("\\", "/").rstrip("/")
    if not (normalized_root == expected_root or normalized_root.endswith(f"/{expected_root}")):
        raise ConfigurationError(f"{collection} output root must end with {expected_root}")

    measurement = _mapping(config, "execution_measurement", MEASUREMENT_KEYS)
    if (
        measurement["profiler_enabled"] is not False
        or measurement["profiled_rounds"] != []
        or measurement["record_cuda_memory"] is not True
        or measurement["record_device_utilization"] is not True
        or measurement["utilization_interval_seconds"] != 2
    ):
        raise ConfigurationError("evaluation measurement configuration is incompatible")
    if config["dataset"]["name"] not in {"shd", "ssc"}:
        raise ConfigurationError("evaluation system supports only SHD and SSC")
    if config["federated"]["clients_per_round"] != 10:
        raise ConfigurationError("evaluation requires ten selected clients per round")


def load_evaluation_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    config = load_resolved_config(config_path)
    validate_evaluation_config(config, repository_root=_repository_root(config_path))
    return config


def _expected_experiments(collection: str):
    return SCHEDULING_EXPERIMENTS if collection == "scheduling_evaluation" else HIERARCHICAL_EXPERIMENTS


def _expected_orders(collection: str) -> dict[int, tuple[str, ...]]:
    if collection == "scheduling_evaluation":
        return {
            37: SCHEDULING_STRATEGIES,
            47: (SCHEDULING_STRATEGIES[1], SCHEDULING_STRATEGIES[2], SCHEDULING_STRATEGIES[0]),
            57: (SCHEDULING_STRATEGIES[2], SCHEDULING_STRATEGIES[0], SCHEDULING_STRATEGIES[1]),
        }
    return {
        37: AGGREGATION_TOPOLOGIES,
        47: (AGGREGATION_TOPOLOGIES[1], AGGREGATION_TOPOLOGIES[0]),
        57: AGGREGATION_TOPOLOGIES,
    }


def _treatment(config: Mapping, collection: str) -> str:
    return (
        config["scheduler"]["strategy"]
        if collection == "scheduling_evaluation"
        else config["aggregation_execution"]["topology"]
    )


def load_evaluation_manifest(path: str | Path) -> list[ManifestTask]:
    manifest_path = Path(path).resolve()
    manifest = _manifest_mapping(manifest_path)
    collection = manifest.get("collection")
    if manifest.get("schema_version") != 1 or collection not in EVALUATION_COLLECTIONS:
        raise ValueError("evaluation manifest schema or collection is incompatible")
    seeds_name = manifest.get("seeds_file")
    if not isinstance(seeds_name, str) or not seeds_name:
        raise ValueError("evaluation manifest requires seeds_file")
    seeds = _manifest_mapping((manifest_path.parent / seeds_name).resolve()).get("seeds")
    if seeds != list(EVALUATION_SEEDS):
        raise ValueError("evaluation seeds must be exactly 37, 47, and 57")
    specifications = _expected_experiments(collection)
    entries = manifest.get("experiments")
    if not isinstance(entries, list) or len(entries) != len(specifications):
        raise ValueError(f"{collection} manifest has an incompatible experiment count")
    repository_root = _repository_root(manifest_path)
    tasks: list[ManifestTask] = []
    templates: dict[tuple[str, str], dict] = {}
    seen_paths: set[Path] = set()
    for specification, entry in zip(specifications, entries, strict=True):
        name, dataset, strategy, topology = specification
        if not isinstance(entry, Mapping) or entry.get("mandatory") is not True or entry.get("id") != name:
            raise ValueError("evaluation manifest experiment order or identity is incompatible")
        configured_path = entry.get("config")
        if not isinstance(configured_path, str) or not configured_path:
            raise ValueError(f"evaluation experiment {name} requires a config path")
        config_path = (manifest_path.parent / configured_path).resolve()
        if config_path in seen_paths or manifest_path.parent not in config_path.parents:
            raise ValueError("evaluation manifest has a duplicate or escaping config path")
        template = load_resolved_config(config_path)
        validate_evaluation_config(template, repository_root=repository_root)
        if (
            template["name"] != name
            or template["metadata"].get("experiment") != name
            or template["dataset"]["name"] != dataset
            or template["scheduler"]["strategy"] != strategy
            or template["aggregation_execution"]["topology"] != topology
            or template["evaluation"]["collection"] != collection
        ):
            raise ValueError("evaluation manifest treatment identity is incompatible")
        templates[(dataset, _treatment(template, collection))] = template
        seen_paths.add(config_path)
        for seed in seeds:
            config = copy.deepcopy(template)
            config["seed"] = seed
            validate_evaluation_config(config, repository_root=repository_root)
            tasks.append(
                ManifestTask(
                    experiment=name,
                    config_path=config_path,
                    seed=seed,
                    dataset=dataset,
                    mode=config["mode"],
                    protocol=config["protocol"],
                    config=config,
                )
            )
    expected_count = 18 if collection == "scheduling_evaluation" else 12
    if len(tasks) != expected_count:
        raise ValueError(f"{collection} must resolve to exactly {expected_count} tasks")
    counts = Counter((task.dataset, _treatment(task.config, collection)) for task in tasks)
    if set(counts.values()) != {3}:
        raise ValueError("every dataset/treatment must contain exactly three evaluation seeds")
    for dataset in ("shd", "ssc"):
        values = [value for (value_dataset, _), value in templates.items() if value_dataset == dataset]
        reference_name = values[0]["evaluation"]["comparison_reference"]
        reference = next((value for value in values if value["name"] == reference_name), None)
        if reference is None:
            raise ValueError(f"{dataset} comparison reference is absent from the resolved treatments")
        for candidate in values:
            validate_resolved_evaluation_pair(reference, candidate, collection)
    names = {task.experiment for task in tasks}
    for task in tasks:
        reference = task.config["evaluation"]["comparison_reference"]
        if reference not in names:
            raise ValueError("evaluation comparison reference is missing")
        if next(value for value in tasks if value.experiment == reference).dataset != task.dataset:
            raise ValueError("evaluation comparison reference crosses datasets")
    paths = [str(_run_directory(task.config)) for task in tasks]
    if len(paths) != len(set(paths)):
        raise ValueError("evaluation manifest contains duplicate run identities or shared output paths")
    return tasks


def load_evaluation_allocations(path: str | Path) -> list[EvaluationAllocation]:
    manifest_path = Path(path).resolve()
    manifest = _manifest_mapping(manifest_path)
    collection = manifest.get("collection")
    tasks = load_evaluation_manifest(manifest_path)
    allocations = manifest.get("allocations")
    if not isinstance(allocations, list) or len(allocations) != 6:
        raise ValueError("evaluation manifest must declare exactly six allocation tasks")
    expected_orders = _expected_orders(collection)
    result = []
    expected_pairs = [(dataset, seed) for dataset in ("shd", "ssc") for seed in EVALUATION_SEEDS]
    for expected_pair, record in zip(expected_pairs, allocations, strict=True):
        dataset, seed = expected_pair
        if (
            not isinstance(record, Mapping)
            or record.get("dataset") != dataset
            or record.get("seed") != seed
            or tuple(record.get("execution_order", ())) != tuple(expected_orders[seed])
        ):
            raise ValueError("evaluation allocation order is incompatible")
        selected_tasks = [task for task in tasks if task.dataset == dataset and task.seed == seed]
        by_treatment = {_treatment(task.config, collection): task for task in selected_tasks}
        ordered_tasks = tuple(by_treatment[value] for value in expected_orders[seed])
        result.append(
            EvaluationAllocation(
                collection=collection,
                dataset=dataset,
                seed=seed,
                execution_order=tuple(expected_orders[seed]),
                tasks=ordered_tasks,
            )
        )
    return result


def validate_collection_path_disjointness(scheduling_manifest: str | Path, hierarchical_manifest: str | Path) -> None:
    scheduling = {str(_run_directory(task.config)) for task in load_evaluation_manifest(scheduling_manifest)}
    hierarchical = {str(_run_directory(task.config)) for task in load_evaluation_manifest(hierarchical_manifest)}
    if scheduling & hierarchical:
        raise ValueError("scheduling and hierarchical evaluation output paths overlap")
