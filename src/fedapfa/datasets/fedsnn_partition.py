"""Deterministic CIFAR-10 partitions following the released Fed-SNN source."""

from __future__ import annotations

import numpy as np

from fedapfa.utilities.serialization import sha256_json

from .dirichlet_partition import DirichletPartition
from .partition_diagnostics import partition_diagnostics


def _artifact(
    *,
    method: str,
    labels: np.ndarray,
    eligible: np.ndarray,
    assigned: list[list[int]],
    seed: int,
    minimum_size: int,
    validation_split_id: str,
    dataset_identity: dict,
    construction_attempts: int,
    alpha: float | None,
) -> DirichletPartition:
    flat = [index for values in assigned for index in values]
    integrity = {
        "complete_assignment": sorted(flat) == sorted(int(value) for value in eligible),
        "unique_assignment": len(flat) == len(set(flat)),
        "minimum_size_satisfied": all(len(values) >= minimum_size for values in assigned),
        "validation_indices_excluded": True,
        "official_test_indices_excluded": True,
    }
    if not all(integrity.values()):
        raise RuntimeError(f"constructed Fed-SNN partition failed integrity checks: {integrity}")
    records, statistics, complete_counts = partition_diagnostics(assigned, labels, eligible)
    clients = []
    for client_index, (indices, diagnostics) in enumerate(zip(assigned, records, strict=True)):
        ordered = sorted(int(value) for value in indices)
        clients.append(
            {
                "client_id": f"client_{client_index:02d}",
                "indices": ordered,
                "size": len(ordered),
                **diagnostics,
            }
        )
    artifact = {
        "schema_version": 2,
        "partition_seed": seed,
        "numpy_random_state_seed": seed % (2**32),
        "validation_split_id": validation_split_id,
        "dataset_identity": dataset_identity,
        "method": method,
        "alpha": alpha,
        "client_count": len(assigned),
        "minimum_examples_per_client": minimum_size,
        "construction_attempts": construction_attempts,
        "eligible_training_examples": len(eligible),
        "complete_eligible_training_class_counts": complete_counts,
        "clients": clients,
        "diagnostic_statistics": statistics,
        "integrity_checks": integrity,
    }
    partition_id = sha256_json(artifact)
    artifact["partition_id"] = partition_id
    return DirichletPartition(partition_id, artifact)


def fedsnn_random_iid_partition(
    *,
    labels: np.ndarray,
    eligible_indices: np.ndarray,
    clients: int,
    minimum_size: int,
    seed: int,
    validation_split_id: str,
    dataset_identity: dict,
) -> DirichletPartition:
    """Assign equal random index sets as in the released ``cifar_iid`` function."""

    labels = np.asarray(labels, dtype=np.int64)
    eligible = np.asarray(eligible_indices, dtype=np.int64)
    if clients <= 0 or len(eligible) % clients:
        raise ValueError("Fed-SNN random IID partition requires an exactly divisible population")
    rng = np.random.RandomState(seed % (2**32))
    remaining = np.sort(eligible.copy())
    per_client = len(eligible) // clients
    assigned: list[list[int]] = []
    for _ in range(clients):
        selected = np.asarray(rng.choice(remaining, per_client, replace=False), dtype=np.int64)
        assigned.append([int(value) for value in selected])
        remaining = np.setdiff1d(remaining, selected, assume_unique=True)
    if len(remaining):
        raise RuntimeError("Fed-SNN random IID partition left eligible indices unassigned")
    return _artifact(
        method="fedsnn_random_iid",
        labels=labels,
        eligible=eligible,
        assigned=assigned,
        seed=seed,
        minimum_size=minimum_size,
        validation_split_id=validation_split_id,
        dataset_identity=dataset_identity,
        construction_attempts=1,
        alpha=None,
    )


def fedsnn_balanced_label_dirichlet_partition(
    *,
    labels: np.ndarray,
    eligible_indices: np.ndarray,
    clients: int,
    alpha: float,
    minimum_size: int,
    seed: int,
    maximum_attempts: int,
    validation_split_id: str,
    dataset_identity: dict,
) -> DirichletPartition:
    """Apply the released class-wise Dirichlet draw and current-size balancing rule."""

    labels = np.asarray(labels, dtype=np.int64)
    eligible = np.asarray(eligible_indices, dtype=np.int64)
    if clients <= 0 or alpha <= 0 or minimum_size <= 0 or maximum_attempts <= 0:
        raise ValueError("invalid Fed-SNN Dirichlet partition settings")
    if len(np.unique(eligible)) != len(eligible):
        raise ValueError("eligible indices must be unique")
    rng = np.random.RandomState(seed % (2**32))
    class_values = np.unique(labels[eligible])
    assigned: list[list[int]] | None = None
    attempts = 0
    target_size = len(eligible) / clients
    for attempt in range(1, maximum_attempts + 1):
        candidate: list[list[int]] = [[] for _ in range(clients)]
        valid_attempt = True
        for class_value in class_values:
            class_indices = eligible[labels[eligible] == class_value].copy()
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.repeat(alpha, clients))
            proportions = np.asarray(
                [
                    proportion * (len(client_indices) < target_size)
                    for proportion, client_indices in zip(proportions, candidate, strict=True)
                ],
                dtype=np.float64,
            )
            if proportions.sum() <= 0:
                valid_attempt = False
                break
            proportions /= proportions.sum()
            boundaries = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            splits = np.split(class_indices, boundaries)
            candidate = [
                client_indices + [int(value) for value in split]
                for client_indices, split in zip(candidate, splits, strict=True)
            ]
        if valid_attempt and min(len(values) for values in candidate) >= minimum_size:
            assigned = candidate
            attempts = attempt
            break
    if assigned is None:
        raise RuntimeError(
            f"unable to construct a Fed-SNN balanced Dirichlet partition after {maximum_attempts} attempts"
        )
    for values in assigned:
        rng.shuffle(values)
    return _artifact(
        method="fedsnn_balanced_label_dirichlet",
        labels=labels,
        eligible=eligible,
        assigned=assigned,
        seed=seed,
        minimum_size=minimum_size,
        validation_split_id=validation_split_id,
        dataset_identity=dataset_identity,
        construction_attempts=attempts,
        alpha=alpha,
    )
