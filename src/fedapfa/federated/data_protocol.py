"""Federated workload roles, partitioning, and lazy official-test access."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import h5py
import numpy as np

from fedapfa.configuration.validation import resolve_dataset_paths
from fedapfa.datasets.centralized_split import stratified_split
from fedapfa.datasets.dirichlet_partition import DirichletPartition, label_dirichlet_partition
from fedapfa.datasets.iid_partition import StratifiedIIDPartition, stratified_iid_partition
from fedapfa.datasets.shd import EventAudioDataset
from fedapfa.datasets.validation import OFFICIAL_EXPECTATIONS, DatasetValidationError
from fedapfa.federated.randomness import resolved_seeds
from fedapfa.utilities.serialization import sha256_json


class OfficialTestAccessError(RuntimeError):
    """The official split was requested before global model selection."""


def file_identity(path: str | Path) -> dict[str, object]:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return {"name": target.name, "size_bytes": target.stat().st_size, "sha256": digest.hexdigest()}


def _labels(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        labels = np.asarray(handle["labels"][:], dtype=np.int64)
    expectation = OFFICIAL_EXPECTATIONS.get(path.name)
    if expectation is not None:
        if len(labels) != expectation.examples:
            raise DatasetValidationError(
                f"{path}: expected {expectation.examples} examples, found {len(labels)}"
            )
        if len(np.unique(labels)) != expectation.classes:
            raise DatasetValidationError(
                f"{path}: expected {expectation.classes} classes in the label collection"
            )
    return labels


@dataclass
class FederatedSHDBundle:
    train_path: Path
    test_path: Path
    config: dict
    labels: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    validation_dataset: EventAudioDataset | None
    partition: DirichletPartition | StratifiedIIDPartition
    split_artifact: dict
    resolved_seed_values: dict[str, int]
    official_test_access_count: int = 0
    official_test_identity: dict | None = None

    @property
    def client_ids(self) -> list[str]:
        return sorted(self.partition.client_indices)

    @property
    def aggregation_weighting(self) -> str:
        return self.config["federated"]["aggregation_weighting"]

    @property
    def checkpoint_selection(self) -> str:
        return self.config["federated"]["checkpoint_selection"]

    @property
    def evaluation_protocol(self) -> dict:
        return {
            "validation_collection": "derived_training_validation",
            "internal_validation_available": True,
            "official_test_publication_collection_name": None,
            "external_implementation_monitors_official_test": False,
            "complete_standard_training_collection": False,
        }

    def client_dataset(self, client_id: str) -> EventAudioDataset:
        indices = self.partition.client_indices[client_id]
        return EventAudioDataset(
            self.train_path,
            indices,
            temporal_bin_ms=self.config["dataset"]["temporal_bin_ms"],
            frequency_bin_factor=self.config["dataset"]["frequency_bin_factor"],
            validate=False,
        )

    def official_test_dataset(self, model_selected: bool) -> EventAudioDataset:
        if not model_selected:
            raise OfficialTestAccessError("official SHD test access is prohibited before global model selection")
        if self.official_test_access_count != 0:
            raise OfficialTestAccessError("official SHD test evaluation is permitted exactly once")
        self.official_test_identity = file_identity(self.test_path)
        self.official_test_access_count += 1
        return EventAudioDataset(
            self.test_path,
            None,
            temporal_bin_ms=self.config["dataset"]["temporal_bin_ms"],
            frequency_bin_factor=self.config["dataset"]["frequency_bin_factor"],
        )


def prepare_federated_shd(config: dict, *, construct_validation: bool = True) -> FederatedSHDBundle:
    paths = resolve_dataset_paths(config)
    train_path = paths["train"]
    test_path = paths["test"]
    labels = _labels(train_path)
    seeds = resolved_seeds(config)
    train_indices, validation_indices = stratified_split(
        labels,
        config["dataset"]["validation_fraction"],
        config["seed"],
    )
    dataset_identity = file_identity(train_path)
    split_core = {
        "schema_version": 1,
        "split_seed": seeds["split"],
        "validation_fraction": config["dataset"]["validation_fraction"],
        "dataset_identity": dataset_identity,
        "training_indices": [int(value) for value in train_indices],
        "validation_indices": [int(value) for value in validation_indices],
    }
    split_artifact = dict(split_core)
    split_artifact["split_id"] = sha256_json(split_core)
    partition_config = config["federated"]["partition"]
    common_partition = {
        "labels": labels,
        "eligible_indices": train_indices,
        "clients": config["federated"]["clients"],
        "minimum_size": partition_config["minimum_examples_per_client"],
        "seed": seeds["partition"],
        "validation_split_id": split_artifact["split_id"],
        "dataset_identity": dataset_identity,
    }
    if partition_config["method"] == "label_dirichlet":
        partition = label_dirichlet_partition(
            **common_partition,
            alpha=partition_config["alpha"],
            maximum_attempts=partition_config["maximum_attempts"],
        )
    elif partition_config["method"] == "stratified_iid":
        partition = stratified_iid_partition(**common_partition)
    else:
        raise ValueError(f"unsupported partition method: {partition_config['method']}")
    validation = (
        EventAudioDataset(
            train_path,
            validation_indices,
            temporal_bin_ms=config["dataset"]["temporal_bin_ms"],
            frequency_bin_factor=config["dataset"]["frequency_bin_factor"],
            validate=False,
        )
        if construct_validation
        else None
    )
    return FederatedSHDBundle(
        train_path=train_path,
        test_path=test_path,
        config=config,
        labels=labels,
        train_indices=train_indices,
        validation_indices=validation_indices,
        validation_dataset=validation,
        partition=partition,
        split_artifact=split_artifact,
        resolved_seed_values=seeds,
    )


@dataclass
class FederatedSSCBundle:
    train_path: Path
    validation_path: Path
    test_path: Path
    config: dict
    labels: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    validation_dataset: EventAudioDataset | None
    partition: DirichletPartition
    split_artifact: dict
    resolved_seed_values: dict[str, int]
    official_test_access_count: int = 0
    official_test_identity: dict | None = None

    @property
    def client_ids(self) -> list[str]:
        return sorted(self.partition.client_indices)

    @property
    def aggregation_weighting(self) -> str:
        return self.config["federated"]["aggregation_weighting"]

    @property
    def checkpoint_selection(self) -> str:
        return self.config["federated"]["checkpoint_selection"]

    @property
    def evaluation_protocol(self) -> dict:
        return {
            "validation_collection": "official_validation",
            "internal_validation_available": False,
            "official_test_publication_collection_name": None,
            "external_implementation_monitors_official_test": False,
            "complete_standard_training_collection": True,
        }

    def client_dataset(self, client_id: str) -> EventAudioDataset:
        return EventAudioDataset(
            self.train_path,
            self.partition.client_indices[client_id],
            temporal_bin_ms=self.config["dataset"]["temporal_bin_ms"],
            frequency_bin_factor=self.config["dataset"]["frequency_bin_factor"],
            validate=False,
        )

    def official_test_dataset(self, model_selected: bool) -> EventAudioDataset:
        if not model_selected:
            raise OfficialTestAccessError("official SSC test access is prohibited before global model selection")
        if self.official_test_access_count != 0:
            raise OfficialTestAccessError("official SSC test evaluation is permitted exactly once")
        self.official_test_identity = file_identity(self.test_path)
        self.official_test_access_count += 1
        return EventAudioDataset(
            self.test_path,
            None,
            temporal_bin_ms=self.config["dataset"]["temporal_bin_ms"],
            frequency_bin_factor=self.config["dataset"]["frequency_bin_factor"],
        )


def prepare_federated_ssc(config: dict, *, construct_validation: bool = True) -> FederatedSSCBundle:
    """Partition complete SSC training data while isolating official evaluation data."""

    paths = resolve_dataset_paths(config)
    train_path = paths["train"]
    validation_path = paths["validation"]
    test_path = paths["test"]
    labels = _labels(train_path)
    validation_labels = _labels(validation_path)
    seeds = resolved_seeds(config)
    train_indices = np.arange(len(labels), dtype=np.int64)
    validation_indices = np.arange(len(validation_labels), dtype=np.int64)
    train_identity = file_identity(train_path)
    validation_identity = file_identity(validation_path)
    split_core = {
        "schema_version": 1,
        "split_seed": seeds["split"],
        "validation_fraction": 0.0,
        "validation_collection": "official_validation",
        "dataset_identity": {
            "name": "ssc",
            "training": train_identity,
            "validation": validation_identity,
        },
        "training_indices": train_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
    }
    split_artifact = dict(split_core)
    split_artifact["split_id"] = sha256_json(split_core)
    partition_config = config["federated"]["partition"]
    partition = label_dirichlet_partition(
        labels=labels,
        eligible_indices=train_indices,
        clients=config["federated"]["clients"],
        alpha=partition_config["alpha"],
        minimum_size=partition_config["minimum_examples_per_client"],
        maximum_attempts=partition_config["maximum_attempts"],
        seed=seeds["partition"],
        validation_split_id=split_artifact["split_id"],
        dataset_identity=train_identity,
    )
    validation = (
        EventAudioDataset(
            validation_path,
            None,
            temporal_bin_ms=config["dataset"]["temporal_bin_ms"],
            frequency_bin_factor=config["dataset"]["frequency_bin_factor"],
        )
        if construct_validation
        else None
    )
    return FederatedSSCBundle(
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        config=config,
        labels=labels,
        train_indices=train_indices,
        validation_indices=validation_indices,
        validation_dataset=validation,
        partition=partition,
        split_artifact=split_artifact,
        resolved_seed_values=seeds,
    )


@runtime_checkable
class FederatedWorkload(Protocol):
    """Dataset-independent boundary consumed by federated coordination."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    validation_dataset: object | None
    partition: DirichletPartition | StratifiedIIDPartition
    split_artifact: dict
    resolved_seed_values: dict[str, int]
    official_test_access_count: int
    client_ids: list[str]
    aggregation_weighting: str
    checkpoint_selection: str
    evaluation_protocol: dict

    def client_dataset(self, client_id: str): ...

    def official_test_dataset(self, model_selected: bool): ...


def prepare_federated_workload(config: dict, *, coordinator: bool) -> FederatedWorkload:
    """Construct client data everywhere and coordinator-owned evaluation data once."""

    dataset_name = config["dataset"]["name"]
    if dataset_name == "shd":
        return prepare_federated_shd(config, construct_validation=coordinator)
    if dataset_name == "ssc":
        return prepare_federated_ssc(config, construct_validation=coordinator)
    if dataset_name == "cifar10":
        from fedapfa.datasets.cifar10 import prepare_federated_cifar10

        return prepare_federated_cifar10(config, construct_validation=coordinator)
    raise ValueError(f"unsupported federated workload dataset: {dataset_name}")
