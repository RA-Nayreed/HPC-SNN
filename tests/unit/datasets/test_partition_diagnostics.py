import math

import numpy as np
import pytest

from fedapfa.datasets.iid_partition import stratified_iid_partition
from fedapfa.datasets.partition_diagnostics import (
    entropy_bits,
    jensen_shannon_divergence_bits,
    normalized_entropy,
)


def _iid(seed=7):
    labels = np.repeat(np.arange(4, dtype=np.int64), 23)
    validation = np.asarray([0, 23, 46, 69], dtype=np.int64)
    official = np.arange(len(labels), len(labels) + 8, dtype=np.int64)
    eligible = np.setdiff1d(np.arange(len(labels), dtype=np.int64), validation)
    result = stratified_iid_partition(
        labels,
        eligible,
        clients=5,
        minimum_size=12,
        seed=seed,
        validation_split_id="split",
        dataset_identity={"name": "synthetic", "sha256": "abc"},
    )
    return labels, validation, official, eligible, result


def test_exact_iid_is_deterministic_balanced_and_excludes_ineligible_indices():
    _, validation, official, eligible, first = _iid()
    *_, second = _iid()
    assert first.artifact == second.artifact
    assigned = [index for values in first.client_indices.values() for index in values]
    assert sorted(assigned) == eligible.tolist()
    assert len(assigned) == len(set(assigned))
    assert set(assigned).isdisjoint(validation)
    assert set(assigned).isdisjoint(official)
    for label in first.artifact["complete_eligible_training_class_counts"]:
        counts = [client["class_counts"][label] for client in first.artifact["clients"]]
        assert max(counts) - min(counts) <= 1
    assert list(first.client_indices) == [f"client_{index:02d}" for index in range(5)]


def test_iid_seed_affects_only_remainder_allocation_and_identity():
    *_, first = _iid(7)
    *_, second = _iid(17)
    assert first.partition_id != second.partition_id
    assert (
        first.artifact["complete_eligible_training_class_counts"]
        == second.artifact["complete_eligible_training_class_counts"]
    )
    assert (
        first.artifact["diagnostic_statistics"]["example_count"]["mean"]
        == second.artifact["diagnostic_statistics"]["example_count"]["mean"]
    )


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ([1, 1], [1, 1], 0.0),
        ([1, 0], [0, 1], 1.0),
        ([0, 0, 5], [0, 0, 5], 0.0),
    ],
)
def test_zero_safe_jensen_shannon_divergence(first, second, expected):
    assert jensen_shannon_divergence_bits(first, second) == pytest.approx(expected)


def test_entropy_and_normalized_entropy_report_missing_labels():
    assert entropy_bits([2, 2, 0, 0]) == pytest.approx(1.0)
    assert normalized_entropy([2, 2, 0, 0]) == pytest.approx(0.5)
    assert math.isfinite(jensen_shannon_divergence_bits([2, 2, 0, 0], [1, 1, 1, 1]))
