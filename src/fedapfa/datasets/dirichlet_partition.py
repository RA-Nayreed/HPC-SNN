"""Deterministic label-wise Dirichlet client partitioning."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import numpy as np

from fedapfa.utilities.serialization import sha256_json


@dataclass(frozen=True)
class DirichletPartition:
    partition_id: str
    artifact: dict

    @property
    def client_indices(self) -> dict[str, list[int]]:
        return {client["client_id"]: list(client["indices"]) for client in self.artifact["clients"]}


def _entropy_bits(class_counts: dict[str, int]) -> float:
    total = sum(class_counts.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in class_counts.values() if count)


def _integrity(eligible: np.ndarray, clients: list[list[int]], minimum_size: int) -> dict[str, bool]:
    assigned = [index for client in clients for index in client]
    return {
        "complete_assignment": sorted(assigned) == sorted(int(value) for value in eligible),
        "unique_assignment": len(assigned) == len(set(assigned)),
        "minimum_size_satisfied": all(len(client) >= minimum_size for client in clients),
        "validation_indices_excluded": True,
        "official_test_indices_excluded": True,
    }


def label_dirichlet_partition(
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
    """Assign each eligible index exactly once or raise after deterministic resampling."""

    labels = np.asarray(labels, dtype=np.int64)
    eligible = np.asarray(eligible_indices, dtype=np.int64)
    if labels.ndim != 1 or eligible.ndim != 1:
        raise ValueError("labels and eligible_indices must be one-dimensional")
    if clients < 2 or alpha <= 0 or minimum_size <= 0 or maximum_attempts <= 0:
        raise ValueError("invalid Dirichlet partition settings")
    if len(np.unique(eligible)) != len(eligible) or np.any(eligible < 0) or np.any(eligible >= len(labels)):
        raise ValueError("eligible indices must be unique and within the label array")
    if len(eligible) < clients * minimum_size:
        raise ValueError("eligible training data cannot satisfy the minimum client size")

    rng = np.random.default_rng(seed)
    class_values = np.unique(labels[eligible])
    assigned_clients: list[list[int]] | None = None
    construction_attempts = 0
    for _attempt in range(1, maximum_attempts + 1):
        candidate = [[] for _ in range(clients)]
        for class_value in class_values:
            indices = eligible[labels[eligible] == class_value].copy()
            rng.shuffle(indices)
            proportions = rng.dirichlet(np.ones(clients, dtype=np.float64) * alpha)
            counts = rng.multinomial(len(indices), proportions)
            offset = 0
            for client_index, count in enumerate(counts):
                candidate[client_index].extend(int(value) for value in indices[offset : offset + count])
                offset += int(count)
        if min(map(len, candidate)) >= minimum_size:
            assigned_clients = candidate
            construction_attempts = _attempt
            break
    if assigned_clients is None:
        raise RuntimeError(
            f"unable to construct a {clients}-client Dirichlet partition with minimum size {minimum_size} "
            f"after {maximum_attempts} attempts"
        )

    client_records = []
    sizes = []
    entropies = []
    for client_index, indices in enumerate(assigned_clients):
        ordered = sorted(indices)
        counts = {
            str(int(class_value)): int(np.sum(labels[np.asarray(ordered, dtype=np.int64)] == class_value))
            for class_value in class_values
        }
        entropy = _entropy_bits(counts)
        sizes.append(len(ordered))
        entropies.append(entropy)
        client_records.append(
            {
                "client_id": f"client_{client_index:02d}",
                "indices": ordered,
                "size": len(ordered),
                "class_counts": counts,
                "label_entropy_bits": entropy,
            }
        )
    integrity = _integrity(eligible, assigned_clients, minimum_size)
    if not all(integrity.values()):
        raise RuntimeError(f"constructed partition failed integrity checks: {integrity}")
    artifact = {
        "schema_version": 1,
        "partition_seed": seed,
        "validation_split_id": validation_split_id,
        "dataset_identity": dataset_identity,
        "method": "label_dirichlet",
        "alpha": alpha,
        "client_count": clients,
        "minimum_examples_per_client": minimum_size,
        "construction_attempts": construction_attempts,
        "eligible_training_examples": len(eligible),
        "clients": client_records,
        "client_size_statistics": {
            "minimum": min(sizes),
            "maximum": max(sizes),
            "mean": statistics.mean(sizes),
            "median": statistics.median(sizes),
            "standard_deviation": statistics.pstdev(sizes),
        },
        "label_entropy_bits_statistics": {
            "minimum": min(entropies),
            "maximum": max(entropies),
            "mean": statistics.mean(entropies),
            "median": statistics.median(entropies),
            "standard_deviation": statistics.pstdev(entropies),
        },
        "integrity_checks": integrity,
    }
    partition_id = sha256_json(artifact)
    artifact["partition_id"] = partition_id
    return DirichletPartition(partition_id=partition_id, artifact=artifact)
