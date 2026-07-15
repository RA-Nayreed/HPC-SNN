import numpy as np

from fedapfa.datasets.centralized_split import select_subset, stratified_split


def test_split_and_subset_are_deterministic():
    labels = np.repeat(np.arange(4), 10)
    assert all(
        np.array_equal(a, b)
        for a, b in zip(stratified_split(labels, 0.2, 7), stratified_split(labels, 0.2, 7), strict=True)
    )
    first = select_subset(labels, 12, 7, True)
    second = select_subset(labels, 12, 7, True)
    assert np.array_equal(first, second) and set(labels[first]) == {0, 1, 2, 3}
