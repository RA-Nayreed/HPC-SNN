"""Deterministic stratified splitting and subset selection."""

import numpy as np


def stratified_split(
    labels: np.ndarray, validation_fraction: float = 0.1, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or not 0 <= validation_fraction < 1:
        raise ValueError("labels must be one-dimensional and validation_fraction in [0, 1)")
    if validation_fraction == 0:
        return np.arange(len(labels), dtype=np.int64), np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * validation_fraction)))
        validation.extend(indices[:count])
        train.extend(indices[count:])
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(validation), dtype=np.int64)


def select_subset(labels: np.ndarray, count: int, seed: int, stratified: bool = True) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if count < 0 or count > len(labels):
        raise ValueError("subset count is outside the dataset")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if not stratified:
        return np.sort(rng.choice(len(labels), count, replace=False))
    classes = np.unique(labels)
    allocation = {int(label): count // len(classes) for label in classes}
    for label in classes[: count % len(classes)]:
        allocation[int(label)] += 1
    selected = []
    for label in classes:
        candidates = np.flatnonzero(labels == label)
        requested = allocation[int(label)]
        selected.extend(rng.choice(candidates, min(requested, len(candidates)), replace=False))
    if len(selected) < count:
        remaining = np.setdiff1d(np.arange(len(labels)), selected)
        selected.extend(rng.choice(remaining, count - len(selected), replace=False))
    return np.asarray(sorted(selected), dtype=np.int64)
