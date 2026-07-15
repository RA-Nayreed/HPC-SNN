"""SHD split isolation and lazy official-test access for FedAvg."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from fedapfa.configuration.validation import resolve_dataset_paths
from fedapfa.datasets.centralized_split import stratified_split
from fedapfa.datasets.dirichlet_partition import DirichletPartition, label_dirichlet_partition
from fedapfa.datasets.shd import EventAudioDataset
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
        return np.asarray(handle["labels"][:], dtype=np.int64)


@dataclass
class FederatedSHDBundle:
    train_path: Path
    test_path: Path
    config: dict
    labels: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    validation_dataset: EventAudioDataset
    partition: DirichletPartition
    split_artifact: dict
    resolved_seed_values: dict[str, int]
    official_test_access_count: int = 0

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
        self.official_test_access_count += 1
        return EventAudioDataset(
            self.test_path,
            None,
            temporal_bin_ms=self.config["dataset"]["temporal_bin_ms"],
            frequency_bin_factor=self.config["dataset"]["frequency_bin_factor"],
        )


def prepare_federated_shd(config: dict) -> FederatedSHDBundle:
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
    partition = label_dirichlet_partition(
        labels=labels,
        eligible_indices=train_indices,
        clients=config["federated"]["clients"],
        alpha=partition_config["alpha"],
        minimum_size=partition_config["minimum_examples_per_client"],
        seed=seeds["partition"],
        maximum_attempts=partition_config["maximum_attempts"],
        validation_split_id=split_artifact["split_id"],
        dataset_identity=dataset_identity,
    )
    validation = EventAudioDataset(
        train_path,
        validation_indices,
        temporal_bin_ms=config["dataset"]["temporal_bin_ms"],
        frequency_bin_factor=config["dataset"]["frequency_bin_factor"],
        validate=False,
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
