import numpy as np
import pytest

from fedapfa.datasets.dirichlet_partition import label_dirichlet_partition


def _partition(seed=91):
    labels = np.tile(np.arange(20, dtype=np.int64), 100)
    validation = np.arange(0, len(labels), 10, dtype=np.int64)
    eligible = np.setdiff1d(np.arange(len(labels), dtype=np.int64), validation)
    result = label_dirichlet_partition(
        labels=labels,
        eligible_indices=eligible,
        clients=20,
        alpha=0.5,
        minimum_size=32,
        seed=seed,
        maximum_attempts=1000,
        validation_split_id="split-identity",
        dataset_identity={"name": "synthetic-shd", "sha256": "abc", "size_bytes": 1},
    )
    return labels, eligible, validation, result


def test_dirichlet_partition_is_deterministic_and_complete():
    _, eligible, _, first = _partition()
    _, _, _, second = _partition()
    assert first.artifact == second.artifact
    assigned = [index for indices in first.client_indices.values() for index in indices]
    assert sorted(assigned) == eligible.tolist()
    assert len(assigned) == len(set(assigned))
    assert first.artifact["integrity_checks"]["complete_assignment"]


def test_dirichlet_partition_excludes_validation_and_meets_minimum_size():
    _, _, validation, result = _partition()
    assigned = {index for indices in result.client_indices.values() for index in indices}
    assert assigned.isdisjoint(set(validation.tolist()))
    assert len(result.client_indices) == 20
    assert min(map(len, result.client_indices.values())) >= 32
    assert all("label_entropy_bits" in client for client in result.artifact["clients"])


def test_dirichlet_partition_rejects_impossible_minimum_size():
    labels = np.arange(40) % 2
    with pytest.raises(ValueError, match="minimum client size"):
        label_dirichlet_partition(
            labels=labels,
            eligible_indices=np.arange(40),
            clients=20,
            alpha=0.5,
            minimum_size=32,
            seed=1,
            maximum_attempts=2,
            validation_split_id="split",
            dataset_identity={"name": "synthetic"},
        )
