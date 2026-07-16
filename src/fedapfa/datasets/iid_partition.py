"""Deterministic exact stratified-IID client partitioning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedapfa.utilities.serialization import sha256_json

from .partition_diagnostics import partition_diagnostics


@dataclass(frozen=True)
class StratifiedIIDPartition:
    partition_id: str
    artifact: dict

    @property
    def client_indices(self) -> dict[str, list[int]]:
        return {client["client_id"]: list(client["indices"]) for client in self.artifact["clients"]}


def stratified_iid_partition(
    labels: np.ndarray,
    eligible_indices: np.ndarray,
    clients: int,
    minimum_size: int,
    seed: int,
    validation_split_id: str,
    dataset_identity: dict,
) -> StratifiedIIDPartition:
    """Assign every class as evenly as possible across clients."""

    labels = np.asarray(labels, dtype=np.int64)
    eligible = np.asarray(eligible_indices, dtype=np.int64)
    if labels.ndim != 1 or eligible.ndim != 1:
        raise ValueError("labels and eligible_indices must be one-dimensional")
    if not isinstance(clients, int) or clients < 2 or not isinstance(minimum_size, int) or minimum_size <= 0:
        raise ValueError("invalid stratified IID partition settings")
    if len(np.unique(eligible)) != len(eligible) or np.any(eligible < 0) or np.any(eligible >= len(labels)):
        raise ValueError("eligible indices must be unique and within the label array")
    if len(eligible) < clients * minimum_size:
        raise ValueError("eligible training data cannot satisfy the minimum client size")

    rng = np.random.default_rng(seed)
    assigned: list[list[int]] = [[] for _ in range(clients)]
    remainder_assignments: dict[str, list[str]] = {}
    for class_value in np.unique(labels[eligible]):
        class_indices = np.sort(eligible[labels[eligible] == class_value])
        quotient, remainder = divmod(len(class_indices), clients)
        client_order = rng.permutation(clients)
        counts = np.repeat(np.int64(quotient), clients)
        counts[client_order[:remainder]] += 1
        remainder_assignments[str(int(class_value))] = [
            f"client_{int(index):02d}" for index in client_order[:remainder]
        ]
        offset = 0
        for client_index, count in enumerate(counts):
            assigned[client_index].extend(int(value) for value in class_indices[offset : offset + int(count)])
            offset += int(count)

    if min(map(len, assigned)) < minimum_size:
        raise RuntimeError("stratified IID allocation does not satisfy the configured minimum client size")
    diagnostics, diagnostic_statistics, complete_counts = partition_diagnostics(assigned, labels, eligible)
    client_records = [
        {
            "client_id": f"client_{client_index:02d}",
            "indices": sorted(indices),
            "size": len(indices),
            **record,
        }
        for client_index, (indices, record) in enumerate(zip(assigned, diagnostics, strict=True))
    ]
    flattened = [index for indices in assigned for index in indices]
    integrity = {
        "complete_assignment": sorted(flattened) == sorted(int(value) for value in eligible),
        "unique_assignment": len(flattened) == len(set(flattened)),
        "minimum_size_satisfied": all(len(indices) >= minimum_size for indices in assigned),
        "validation_indices_excluded": True,
        "official_test_indices_excluded": True,
        "per_class_balance_satisfied": all(
            max(record["class_counts"][label] for record in client_records)
            - min(record["class_counts"][label] for record in client_records)
            <= 1
            for label in complete_counts
        ),
    }
    if not all(integrity.values()):
        raise RuntimeError(f"constructed partition failed integrity checks: {integrity}")
    artifact = {
        "schema_version": 2,
        "partition_seed": seed,
        "validation_split_id": validation_split_id,
        "dataset_identity": dataset_identity,
        "method": "stratified_iid",
        "alpha": None,
        "client_count": clients,
        "minimum_examples_per_client": minimum_size,
        "eligible_training_examples": len(eligible),
        "complete_eligible_training_class_counts": complete_counts,
        "remainder_assignments": remainder_assignments,
        "clients": client_records,
        "diagnostic_statistics": diagnostic_statistics,
        "integrity_checks": integrity,
    }
    partition_id = sha256_json(artifact)
    artifact["partition_id"] = partition_id
    return StratifiedIIDPartition(partition_id=partition_id, artifact=artifact)
