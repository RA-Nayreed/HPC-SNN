"""Dependency-free deterministic classification metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def confusion_matrix(
    predictions: Sequence[int] | torch.Tensor,
    targets: Sequence[int] | torch.Tensor,
    classes: int,
) -> list[list[int]]:
    """Count target rows and predicted columns."""

    if not isinstance(classes, int) or classes <= 0:
        raise ValueError("classes must be a positive integer")
    predicted = torch.as_tensor(predictions, dtype=torch.int64).reshape(-1)
    expected = torch.as_tensor(targets, dtype=torch.int64).reshape(-1)
    if predicted.numel() != expected.numel():
        raise ValueError("predictions and targets must have equal length")
    if predicted.numel() and (
        bool(torch.any(predicted < 0))
        or bool(torch.any(predicted >= classes))
        or bool(torch.any(expected < 0))
        or bool(torch.any(expected >= classes))
    ):
        raise ValueError("classification labels are outside the configured class range")
    encoded = expected * classes + predicted
    matrix = torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes)
    return [[int(value) for value in row] for row in matrix]


def per_class_accuracy(matrix: Sequence[Sequence[int]]) -> list[float]:
    """Return per-target-class accuracy, using zero for absent classes."""

    result = []
    for index, row in enumerate(matrix):
        total = sum(int(value) for value in row)
        result.append(0.0 if total == 0 else int(row[index]) / total)
    return result


def macro_f1_from_confusion_matrix(matrix: Sequence[Sequence[int]]) -> float:
    """Return macro-F1 with zero for undefined precision or recall."""

    rows = [list(map(int, row)) for row in matrix]
    if not rows or any(len(row) != len(rows) for row in rows):
        raise ValueError("confusion matrix must be nonempty and square")
    scores = []
    for index in range(len(rows)):
        true_positive = rows[index][index]
        predicted_total = sum(row[index] for row in rows)
        target_total = sum(rows[index])
        precision = 0.0 if predicted_total == 0 else true_positive / predicted_total
        recall = 0.0 if target_total == 0 else true_positive / target_total
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    result = sum(scores) / len(scores)
    if not math.isfinite(result):
        raise ValueError("macro-F1 is not finite")
    return result
