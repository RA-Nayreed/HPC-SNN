"""Strict configuration for scaling/energy and non-IID/energy evaluations."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path

from fedapfa.scheduling.base import ASSIGNMENT_TIE_BREAKING_VERSION, EVENT_STRUCTURE_FEATURES

from .evaluation_system import AGGREGATION_KEYS, PARALLEL_KEYS, SCHEDULER_KEYS, stable_rank_mapping
from .experiment_id import experiment_id
from .federated_validation import validate_federated_config
from .loader import load_resolved_config
from .manifest import ManifestTask, _manifest_mapping
from .validation import ConfigurationError

COMPARATIVE_SEEDS = (37, 47, 57)
COMPARATIVE_COLLECTIONS = (
    "system_scaling_energy_evaluation",
    "non_iid_energy_evaluation",
)
SCALING_TOPOLOGIES = {
    "one_node_one_gpu": (1, 1, 1, 1),
    "one_node_two_gpu": (1, 2, 2, 2),
    "one_node_four_gpu": (1, 4, 4, 4),
    "two_nodes_four_gpus": (2, 2, 4, 4),
}
NON_IID_TREATMENTS = {
    "iid": ("stratified_iid", None),
    "dirichlet_alpha_1_0": ("label_dirichlet", 1.0),
    "dirichlet_alpha_0_5": ("label_dirichlet", 0.5),
    "dirichlet_alpha_0_1": ("label_dirichlet", 0.1),
}
NON_IID_EXECUTION_ORDERS = {
    37: ("iid", "dirichlet_alpha_1_0", "dirichlet_alpha_0_5", "dirichlet_alpha_0_1"),
    47: ("dirichlet_alpha_1_0", "dirichlet_alpha_0_1", "iid", "dirichlet_alpha_0_5"),
    57: ("dirichlet_alpha_0_5", "iid", "dirichlet_alpha_0_1", "dirichlet_alpha_1_0"),
}

SCALING_EXPERIMENTS = tuple(
    (f"{dataset}_{model}_system_scaling_{topology}", dataset, topology)
    for dataset, model in (("shd", "lif"), ("ssc", "lif_128"))
    for topology in SCALING_TOPOLOGIES
)
NON_IID_EXPERIMENTS = tuple(
    (f"{dataset}_{model}_non_iid_{treatment}", dataset, treatment)
    for dataset, model in (("shd", "lif"), ("ssc", "lif_128"))
    for treatment in NON_IID_TREATMENTS
)

COMPARATIVE_EVALUATION_KEYS = {
    "collection",
    "treatment_id",
    "comparison_reference",
    "allocation_grouping",
    "execution_order_design",
    "resolved_configuration_whitelist_version",
    "evidence_complete_outcome",
}
ENERGY_MEASUREMENT_KEYS = {
    "sampling_backend",
    "sampling_interval_ms",
    "maximum_sample_gap_ms",
    "idle_before_seconds",
    "idle_after_seconds",
    "boundary_reconciliation_tolerance_joules",
    "node_file_schema_version",
    "require_leading_and_trailing_samples",
    "require_cumulative_energy_crosscheck_when_supported",
}
CALIBRATION_REQUIREMENT_KEYS = {
    "required",
    "paired_repetitions",
    "alternating_order",
    "exclude_warmups",
    "maximum_median_runtime_overhead_fraction",
    "minimum_interval_coverage_fraction",
    "require_update_identity",
    "official_test_access_count",
    "node_count",
    "device_count",
    "process_count",
    "sampling_interval_ms",
    "sampler_topology",
}
FROZEN_MODEL_KEYS = {
    "model_path",
    "model_sha256",
    "provenance_path",
    "provenance_sha256",
    "feature_order",
    "fitting_seeds",
    "evaluation_seed",
    "target",
    "use_for_assignment",
    "evaluation_role",
}
FROZEN_DIAGNOSTIC_KEYS = {"runtime", "gross_energy"}

RESOLVED_COMPARISON_WHITELIST_VERSION = "resolved_leaf_paths_v1"
FROZEN_MODEL_FITTING_ROW_IDENTITY_SHA256 = "3f46ac7872fd3e1987a3810467a4a819a9226c21e3632a3858e7335a58e49695"
FROZEN_PROVENANCE_FITTING_ROW_IDENTITY_SHA256 = "8d5c11bb72e34405965062f2d32eb98a98e812130992ac7a409c88e27b9557d3"
FROZEN_PROVENANCE_VALIDATION_ROW_IDENTITY_SHA256 = "fb9683856dde589f5d90b7b3dc565ae4b643d16e289cae6149844911a2feb21b"
FROZEN_PROVENANCE_EVALUATION_ROW_IDENTITY_SHA256 = "6e1d858ef00609493df1c38ef2ffd3370d7cef08294832371f37faea3edf0cf3"
_IDENTITY_DIFFERENCES = {
    "name",
    "metadata.experiment",
    "comparative_evaluation.treatment_id",
}
_SCALING_PHYSICAL_DIFFERENCES = {
    "parallel_execution.node_count",
    "parallel_execution.devices_per_node",
    "parallel_execution.device_count",
    "parallel_execution.process_count",
    "calibration_requirements.node_count",
    "calibration_requirements.device_count",
    "calibration_requirements.process_count",
    "calibration_requirements.sampler_topology",
}
_NON_IID_DISTRIBUTION_DIFFERENCES = {
    "federated.partition.method",
    "federated.partition.alpha",
}
RESOLVED_COMPARISON_ALLOWED_DIFFERENCES = {
    "system_scaling_energy_evaluation": _IDENTITY_DIFFERENCES | _SCALING_PHYSICAL_DIFFERENCES,
    "non_iid_energy_evaluation": _IDENTITY_DIFFERENCES | _NON_IID_DISTRIBUTION_DIFFERENCES,
}


@dataclass(frozen=True)
class ComparativeAllocation:
    collection: str
    dataset: str
    seed: int
    allocation_index: int
    execution_order: tuple[str, ...]
    tasks: tuple[ManifestTask, ...]


def _mapping(config: Mapping, name: str, keys: set[str]) -> Mapping:
    value = config.get(name)
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ConfigurationError(f"{name} must contain exactly {sorted(keys)}")
    return value


def _positive(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f"{label} must be finite and positive")
    return float(value)


def _nonnegative(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ConfigurationError(f"{label} must be finite and nonnegative")
    return float(value)


def _repository_root(path: Path) -> Path:
    for candidate in (path.resolve(), *path.resolve().parents):
        if (candidate / ".git").exists() and (candidate / "results").is_dir():
            return candidate
    raise ValueError(f"repository root could not be resolved from {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@cache
def _validate_frozen_artifact_files(
    model_path_text: str,
    model_sha256: str,
    provenance_path_text: str,
    provenance_sha256: str,
    feature_order: tuple[str, ...],
    fitting_seeds: tuple[int, ...],
    evaluation_seed: int,
) -> None:
    model_path = Path(model_path_text)
    provenance_path = Path(provenance_path_text)
    if _sha256(model_path) != model_sha256:
        raise ConfigurationError(f"frozen diagnostic SHA-256 differs: {model_path}")
    if _sha256(provenance_path) != provenance_sha256:
        raise ConfigurationError(f"frozen diagnostic SHA-256 differs: {provenance_path}")
    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    if artifact.get("feature_order") != list(feature_order):
        raise ConfigurationError("frozen diagnostic artifact feature order differs")
    artifact_fitting_seeds = sorted({int(value) for value in artifact.get("fitting_seeds", [])})
    evaluation_seeds = sorted({int(value) for value in artifact.get("evaluation_seeds", [])})
    if artifact_fitting_seeds and artifact_fitting_seeds != list(fitting_seeds):
        raise ConfigurationError("frozen diagnostic artifact fitting seeds differ")
    if evaluation_seeds and evaluation_seeds != [evaluation_seed]:
        raise ConfigurationError("frozen diagnostic artifact evaluation seed differs")
    if (
        len(artifact.get("fitting_row_hashes", [])) != 4000
        or _sha256_json(artifact.get("fitting_row_hashes")) != FROZEN_MODEL_FITTING_ROW_IDENTITY_SHA256
    ):
        raise ConfigurationError("frozen diagnostic fitting-row identities differ")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    joint = provenance.get("settings", {}).get("joint", {})
    expected_rows = {
        "fitting_row_identities": (2988, FROZEN_PROVENANCE_FITTING_ROW_IDENTITY_SHA256),
        "validation_row_identities": (
            1012,
            FROZEN_PROVENANCE_VALIDATION_ROW_IDENTITY_SHA256,
        ),
        "evaluation_row_identities": (
            2000,
            FROZEN_PROVENANCE_EVALUATION_ROW_IDENTITY_SHA256,
        ),
    }
    for field, (count, expected_sha256) in expected_rows.items():
        if len(joint.get(field, [])) != count or _sha256_json(joint.get(field)) != expected_sha256:
            raise ConfigurationError(f"frozen diagnostic {field.replace('_', '-')} differ")


def _leaf_differences(first, second, prefix: str = "") -> dict[str, tuple[object, object]]:
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        result = {}
        for key in sorted(set(first) | set(second)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in first or key not in second:
                result[path] = (copy.deepcopy(first.get(key)), copy.deepcopy(second.get(key)))
            else:
                result.update(_leaf_differences(first[key], second[key], path))
        return result
    return {} if first == second else {prefix: (copy.deepcopy(first), copy.deepcopy(second))}


def comparative_scientific_identity(config: Mapping) -> dict:
    """Return the scientific identity, excluding execution placement and evidence paths."""

    identity = copy.deepcopy(dict(config))
    for key in (
        "name",
        "output_root",
        "resume",
        "metadata",
        "pairing_group",
        "parallel_execution",
        "execution_measurement",
        "energy_measurement",
        "calibration_requirements",
        "instrumentation_calibration_identity",
        "comparative_evaluation",
    ):
        identity.pop(key, None)
    return identity


def comparative_execution_identity(config: Mapping) -> dict:
    """Return topology, measurement, scheduling, and aggregation resume identity."""

    parallel = copy.deepcopy(dict(config["parallel_execution"]))
    return {
        "parallel_execution": parallel,
        "process_to_device_mapping": stable_rank_mapping(
            int(parallel["node_count"]), int(parallel["devices_per_node"])
        ),
        "scheduler": copy.deepcopy(dict(config["scheduler"])),
        "aggregation_execution": copy.deepcopy(dict(config["aggregation_execution"])),
        "execution_measurement": copy.deepcopy(dict(config["execution_measurement"])),
        "energy_measurement": copy.deepcopy(dict(config["energy_measurement"])),
        "calibration_requirements": copy.deepcopy(dict(config["calibration_requirements"])),
        "instrumentation_calibration_identity": copy.deepcopy(config.get("instrumentation_calibration_identity")),
        "comparative_evaluation": copy.deepcopy(dict(config["comparative_evaluation"])),
    }


def validate_resolved_comparative_pair(first: Mapping, second: Mapping, collection: str) -> dict:
    """Compare fully resolved mappings using the versioned collection whitelist."""

    if collection not in RESOLVED_COMPARISON_ALLOWED_DIFFERENCES:
        raise ConfigurationError("resolved comparison collection is unsupported")
    first_collection = first.get("comparative_evaluation", {}).get("collection")
    second_collection = second.get("comparative_evaluation", {}).get("collection")
    if first_collection != collection or second_collection != collection:
        raise ConfigurationError("resolved comparison crosses collections")
    differences = _leaf_differences(first, second)
    unexpected = sorted(set(differences) - RESOLVED_COMPARISON_ALLOWED_DIFFERENCES[collection])
    if unexpected:
        raise ConfigurationError(f"resolved paired configurations differ outside the whitelist: {unexpected}")
    return {
        "whitelist_version": RESOLVED_COMPARISON_WHITELIST_VERSION,
        "collection": collection,
        "allowed_difference_paths": sorted(RESOLVED_COMPARISON_ALLOWED_DIFFERENCES[collection]),
        "observed_difference_paths": sorted(differences),
    }


def _validate_frozen_models(config: Mapping, repository_root: Path | None) -> None:
    diagnostics = _mapping(config, "frozen_model_diagnostics", FROZEN_DIAGNOSTIC_KEYS)
    for name, record in diagnostics.items():
        if not isinstance(record, Mapping) or set(record) != FROZEN_MODEL_KEYS:
            raise ConfigurationError(f"frozen_model_diagnostics.{name} is malformed")
        if record["feature_order"] != list(EVENT_STRUCTURE_FEATURES):
            raise ConfigurationError(f"frozen {name} diagnostic feature order is incompatible")
        if record["fitting_seeds"] != [7, 17] or record["evaluation_seed"] != 27:
            raise ConfigurationError(f"frozen {name} diagnostic seed provenance is incompatible")
        if record["use_for_assignment"] is not False or record["evaluation_role"] != "transfer_evaluation_only":
            raise ConfigurationError("frozen models are diagnostics only and cannot assign clients")
        for field in ("model_path", "provenance_path", "model_sha256", "provenance_sha256", "target"):
            if not isinstance(record[field], str) or not record[field]:
                raise ConfigurationError(f"frozen {name} diagnostic {field} is missing")
        if repository_root is not None:
            for path_field in ("model_path", "provenance_path"):
                path = (repository_root / record[path_field]).resolve()
                if repository_root not in path.parents or not path.is_file():
                    raise ConfigurationError(f"frozen diagnostic artifact is unavailable: {path}")
            _validate_frozen_artifact_files(
                str((repository_root / record["model_path"]).resolve()),
                record["model_sha256"],
                str((repository_root / record["provenance_path"]).resolve()),
                record["provenance_sha256"],
                tuple(record["feature_order"]),
                tuple(record["fitting_seeds"]),
                int(record["evaluation_seed"]),
            )


def validate_comparative_evaluation_config(
    config: Mapping,
    *,
    repository_root: str | Path | None = None,
) -> None:
    """Reject scientific, topology, scheduler, aggregation, and measurement drift."""

    validate_federated_config(config)
    root = None if repository_root is None else Path(repository_root).resolve()
    evaluation = _mapping(config, "comparative_evaluation", COMPARATIVE_EVALUATION_KEYS)
    collection = evaluation["collection"]
    if collection not in COMPARATIVE_COLLECTIONS:
        raise ConfigurationError("comparative evaluation collection is unsupported")
    if evaluation["resolved_configuration_whitelist_version"] != RESOLVED_COMPARISON_WHITELIST_VERSION:
        raise ConfigurationError("resolved-configuration whitelist version is incompatible")
    if evaluation["execution_order_design"] != (
        "independent_allocations" if collection == COMPARATIVE_COLLECTIONS[0] else "rotated_order"
    ):
        raise ConfigurationError("comparative execution-order design is incompatible")
    if evaluation["allocation_grouping"] != (
        "one_execution_per_allocation"
        if collection == COMPARATIVE_COLLECTIONS[0]
        else "four_sequential_executions_per_allocation"
    ):
        raise ConfigurationError("comparative allocation grouping is incompatible")
    expected_outcome = (
        "system_scaling_energy_characterization_complete"
        if collection == COMPARATIVE_COLLECTIONS[0]
        else "non_iid_energy_characterization_complete"
    )
    if evaluation["evidence_complete_outcome"] != expected_outcome:
        raise ConfigurationError("evidence-complete outcome label is incompatible")

    scheduler = _mapping(config, "scheduler", SCHEDULER_KEYS)
    if (
        scheduler["strategy"] != "example_count_longest_processing_time"
        or scheduler["tie_breaking_version"] != ASSIGNMENT_TIE_BREAKING_VERSION
        or scheduler["predictions_before_client_training"] is not True
        or scheduler["training_data_only"] is not True
    ):
        raise ConfigurationError(
            "comparative evaluation requires fixed example-count longest-processing-time scheduling"
        )
    cost_model = scheduler["cost_model"]
    if not isinstance(cost_model, Mapping) or cost_model.get("feature_order") != list(EVENT_STRUCTURE_FEATURES):
        raise ConfigurationError("scheduler frozen diagnostic provenance is malformed")

    aggregation = _mapping(config, "aggregation_execution", AGGREGATION_KEYS)
    if (
        aggregation["topology"] != "flat_ordered"
        or aggregation["order_policy"] != "selected_client_order"
        or aggregation["accumulator_dtype"] != "float64"
    ):
        raise ConfigurationError("flat ordered selected-client aggregation is authoritative")
    _nonnegative(aggregation["absolute_tolerance"], "aggregation absolute tolerance")
    _nonnegative(aggregation["relative_tolerance"], "aggregation relative tolerance")
    _nonnegative(aggregation["material_runtime_regression_fraction"], "material runtime fraction")

    parallel = _mapping(config, "parallel_execution", PARALLEL_KEYS)
    topology = (
        parallel["node_count"],
        parallel["devices_per_node"],
        parallel["device_count"],
        parallel["process_count"],
    )
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in topology):
        raise ConfigurationError("parallel topology values must be positive integers")
    if topology[2] != topology[0] * topology[1] or topology[3] != topology[2]:
        raise ConfigurationError("parallel topology requires one process per physical GPU")
    if (
        parallel["client_processes_per_device"] != 1
        or parallel["control_backend"] != "nccl"
        or parallel["cuda_process_service"] != "none"
        or parallel["client_assignment"] != "example_count_longest_processing_time"
        or parallel["aggregation_order"] != "selected_client_order"
        or parallel["aggregation_topology"] != "flat_ordered"
        or parallel["rank_mapping"] != "node_major_local_rank"
        or parallel["synchronize_each_round"] is not True
    ):
        raise ConfigurationError("parallel execution must use exclusive NCCL ranks and flat ordered aggregation")

    treatment = evaluation["treatment_id"]
    partition = config["federated"]["partition"]
    if collection == COMPARATIVE_COLLECTIONS[0]:
        if treatment not in SCALING_TOPOLOGIES or topology != SCALING_TOPOLOGIES[treatment]:
            raise ConfigurationError("scaling treatment and physical topology differ")
        if partition.get("method") != "label_dirichlet" or partition.get("alpha") != 0.5:
            raise ConfigurationError("scaling evaluation requires label-Dirichlet alpha 0.5")
        expected_root = "runs/system_scaling_energy_evaluation"
    else:
        if topology != SCALING_TOPOLOGIES["one_node_four_gpu"]:
            raise ConfigurationError("non-IID evaluation requires one node and four physical GPUs")
        if (
            treatment not in NON_IID_TREATMENTS
            or (partition.get("method"), partition.get("alpha")) != NON_IID_TREATMENTS[treatment]
        ):
            raise ConfigurationError("non-IID treatment and partition settings differ")
        expected_root = "runs/non_iid_energy_evaluation"
    normalized_root = str(config["output_root"]).replace("\\", "/").rstrip("/")
    if normalized_root != expected_root and not normalized_root.endswith(f"/{expected_root}"):
        raise ConfigurationError(f"comparative output root must end with {expected_root}")

    execution_measurement = config.get("execution_measurement")
    expected_execution_measurement = {
        "profiler_enabled": False,
        "profiled_rounds": [],
        "record_cuda_memory": True,
        "record_device_utilization": True,
        "utilization_interval_seconds": 0.1,
    }
    if execution_measurement != expected_execution_measurement:
        raise ConfigurationError("comparative execution measurement settings are incompatible")
    energy = _mapping(config, "energy_measurement", ENERGY_MEASUREMENT_KEYS)
    if (
        energy["sampling_backend"] != "nvml"
        or energy["sampling_interval_ms"] != 100
        or energy["maximum_sample_gap_ms"] != 250
        or energy["node_file_schema_version"] != 1
        or energy["require_leading_and_trailing_samples"] is not True
        or energy["require_cumulative_energy_crosscheck_when_supported"] is not True
    ):
        raise ConfigurationError("energy sampling must use validated 100 ms NVML telemetry with 250 ms gap rejection")
    _positive(energy["idle_before_seconds"], "idle-before duration")
    _positive(energy["idle_after_seconds"], "idle-after duration")
    _nonnegative(energy["boundary_reconciliation_tolerance_joules"], "energy boundary tolerance")

    calibration = _mapping(config, "calibration_requirements", CALIBRATION_REQUIREMENT_KEYS)
    expected_calibration = {
        "required": True,
        "paired_repetitions": 10,
        "alternating_order": True,
        "exclude_warmups": True,
        "maximum_median_runtime_overhead_fraction": 0.02,
        "minimum_interval_coverage_fraction": 0.9,
        "require_update_identity": True,
        "official_test_access_count": 0,
        "node_count": topology[0],
        "device_count": topology[2],
        "process_count": topology[3],
        "sampling_interval_ms": 100,
        "sampler_topology": f"{topology[0]}_node_{topology[2]}_device_node_local",
    }
    if calibration != expected_calibration:
        raise ConfigurationError("instrumentation calibration requirements differ from the execution topology")
    _validate_frozen_models(config, root)

    federation = config["federated"]
    batch_size = 32 if config["dataset"]["name"] == "shd" else 256
    required_workload = {
        "clients": 20,
        "clients_per_round": 10,
        "rounds": 100,
        "local_epochs": 1,
        "local_batch_size": batch_size,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "aggregation_weighting": "example_count",
        "checkpoint_selection": "best_validation",
    }
    for key, expected in required_workload.items():
        if federation.get(key) != expected:
            raise ConfigurationError(f"comparative workload requires federated.{key}={expected!r}")
    if config.get("metadata", {}).get("experiment") != config.get("name"):
        raise ConfigurationError("metadata experiment must equal the execution name")


def load_comparative_evaluation_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    config = load_resolved_config(config_path)
    validate_comparative_evaluation_config(config, repository_root=_repository_root(config_path))
    return config


def _specifications(collection: str):
    return SCALING_EXPERIMENTS if collection == COMPARATIVE_COLLECTIONS[0] else NON_IID_EXPERIMENTS


def _run_directory(config: Mapping) -> Path:
    return Path(config["output_root"]) / experiment_id(dict(config))


def load_comparative_evaluation_manifest(path: str | Path) -> list[ManifestTask]:
    manifest_path = Path(path).resolve()
    manifest = _manifest_mapping(manifest_path)
    collection = manifest.get("collection")
    if manifest.get("schema_version") != 1 or collection not in COMPARATIVE_COLLECTIONS:
        raise ValueError("comparative manifest schema or collection is incompatible")
    seeds_file = manifest.get("seeds_file")
    if not isinstance(seeds_file, str) or not seeds_file:
        raise ValueError("comparative manifest requires seeds_file")
    seeds = _manifest_mapping((manifest_path.parent / seeds_file).resolve()).get("seeds")
    if seeds != list(COMPARATIVE_SEEDS):
        raise ValueError("comparative seeds must be exactly 37, 47, and 57")
    specifications = _specifications(collection)
    entries = manifest.get("experiments")
    if not isinstance(entries, list) or len(entries) != len(specifications):
        raise ValueError("comparative manifest must contain exactly eight executable treatments")
    root = _repository_root(manifest_path)
    tasks = []
    templates: dict[tuple[str, str], dict] = {}
    seen_paths = set()
    for specification, entry in zip(specifications, entries, strict=True):
        name, dataset, treatment = specification
        if not isinstance(entry, Mapping) or entry.get("id") != name or entry.get("mandatory") is not True:
            raise ValueError("comparative manifest experiment order or identity is incompatible")
        config_name = entry.get("config")
        if not isinstance(config_name, str) or not config_name:
            raise ValueError(f"comparative experiment {name} requires a config path")
        config_path = (manifest_path.parent / config_name).resolve()
        if manifest_path.parent not in config_path.parents or config_path in seen_paths:
            raise ValueError("comparative manifest config path is duplicate or escapes its collection")
        template = load_resolved_config(config_path)
        validate_comparative_evaluation_config(template, repository_root=root)
        if (
            template["name"] != name
            or template["metadata"]["experiment"] != name
            or template["dataset"]["name"] != dataset
            or template["comparative_evaluation"]["collection"] != collection
            or template["comparative_evaluation"]["treatment_id"] != treatment
        ):
            raise ValueError("comparative manifest treatment identity is incompatible")
        templates[(dataset, treatment)] = template
        seen_paths.add(config_path)
        for seed in seeds:
            config = copy.deepcopy(template)
            config["seed"] = seed
            validate_comparative_evaluation_config(config, repository_root=root)
            tasks.append(ManifestTask(name, config_path, seed, dataset, config["mode"], config["protocol"], config))
    if len(tasks) != 24:
        raise ValueError(f"{collection} must expand to exactly 24 executions")
    counts = Counter((task.dataset, task.config["comparative_evaluation"]["treatment_id"]) for task in tasks)
    if len(counts) != 8 or set(counts.values()) != {3}:
        raise ValueError("every comparative dataset/treatment must contain three seeds")
    for dataset in ("shd", "ssc"):
        dataset_templates = {key[1]: value for key, value in templates.items() if key[0] == dataset}
        reference = next(
            (
                value
                for value in dataset_templates.values()
                if value["name"] == value["comparative_evaluation"]["comparison_reference"]
            ),
            None,
        )
        if reference is None:
            raise ValueError(f"{dataset} comparison reference is missing")
        for candidate in dataset_templates.values():
            validate_resolved_comparative_pair(reference, candidate, collection)
    paths = [str(_run_directory(task.config)) for task in tasks]
    if len(paths) != len(set(paths)):
        raise ValueError("comparative manifest contains duplicate run paths")
    return tasks


def validate_resolved_comparative_manifest(
    path: str | Path,
    *,
    data_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> list[dict]:
    """Validate every dataset/seed comparison after runtime path overrides."""

    manifest_path = Path(path).resolve()
    root = _repository_root(manifest_path)
    tasks = load_comparative_evaluation_manifest(manifest_path)
    collection = tasks[0].config["comparative_evaluation"]["collection"]
    records = []
    for dataset in ("shd", "ssc"):
        for seed in COMPARATIVE_SEEDS:
            selected = [task for task in tasks if task.dataset == dataset and task.seed == seed]
            resolved = []
            for task in selected:
                config = copy.deepcopy(task.config)
                if data_root is not None:
                    config["dataset"]["root"] = str(data_root)
                if output_root is not None:
                    config["output_root"] = str(output_root)
                validate_comparative_evaluation_config(config, repository_root=root)
                resolved.append(config)
            reference_name = resolved[0]["comparative_evaluation"]["comparison_reference"]
            reference = next((value for value in resolved if value["name"] == reference_name), None)
            if reference is None:
                raise ConfigurationError(f"{dataset} seed {seed} comparison reference is missing")
            records.extend(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "treatment_id": value["comparative_evaluation"]["treatment_id"],
                    **validate_resolved_comparative_pair(reference, value, collection),
                }
                for value in resolved
            )
    if len(records) != 24:
        raise ConfigurationError("resolved comparative manifest must contain exactly 24 pair records")
    return records


def load_comparative_allocations(path: str | Path) -> list[ComparativeAllocation]:
    manifest_path = Path(path).resolve()
    manifest = _manifest_mapping(manifest_path)
    collection = manifest.get("collection")
    tasks = load_comparative_evaluation_manifest(manifest_path)
    records = manifest.get("allocations")
    expected_count = 24 if collection == COMPARATIVE_COLLECTIONS[0] else 6
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError(f"{collection} must declare exactly {expected_count} allocations")
    by_key = {(task.dataset, task.seed, task.config["comparative_evaluation"]["treatment_id"]): task for task in tasks}
    allocations = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("allocation_index") != index:
            raise ValueError("comparative allocation indices must be contiguous and ordered")
        dataset = record.get("dataset")
        seed = record.get("seed")
        order = tuple(record.get("execution_order", ()))
        expected_order = (
            (record.get("topology"),)
            if collection == COMPARATIVE_COLLECTIONS[0]
            else NON_IID_EXECUTION_ORDERS.get(seed)
        )
        if dataset not in {"shd", "ssc"} or seed not in COMPARATIVE_SEEDS or order != expected_order:
            raise ValueError("comparative allocation dataset, seed, or execution order is incompatible")
        selected = tuple(by_key[(dataset, seed, treatment)] for treatment in order)
        selected = tuple(
            replace(
                task,
                config={
                    **copy.deepcopy(task.config),
                    "allocation_provenance": {
                        "allocation_index": index,
                        "execution_order": list(order),
                        "treatment_position": position,
                        "order_design": (
                            "independent_allocations" if collection == COMPARATIVE_COLLECTIONS[0] else "rotated_order"
                        ),
                    },
                },
            )
            for position, task in enumerate(selected, start=1)
        )
        allocations.append(ComparativeAllocation(collection, dataset, seed, index, order, selected))
    if {task.experiment for allocation in allocations for task in allocation.tasks} != {
        task.experiment for task in tasks
    }:
        raise ValueError("comparative allocations do not cover every treatment")
    if len([task for allocation in allocations for task in allocation.tasks]) != 24:
        raise ValueError("comparative allocations must cover exactly 24 executions")
    return allocations


def validate_comparative_path_disjointness(first: str | Path, second: str | Path) -> None:
    first_paths = {_run_directory(task.config) for task in load_comparative_evaluation_manifest(first)}
    second_paths = {_run_directory(task.config) for task in load_comparative_evaluation_manifest(second)}
    if first_paths & second_paths:
        raise ValueError("scaling and non-IID comparative run paths overlap")
