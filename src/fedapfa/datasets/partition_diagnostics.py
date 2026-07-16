"""Deterministic label-distribution diagnostics for client partitions."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence


def _probabilities(counts: Sequence[int | float]) -> list[float]:
    values = [float(value) for value in counts]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("class counts must be finite and nonnegative")
    total = sum(values)
    return [0.0 for _ in values] if total == 0 else [value / total for value in values]


def entropy_bits(counts: Sequence[int | float]) -> float:
    """Return Shannon entropy in bits with zero-count classes ignored."""

    return -sum(probability * math.log2(probability) for probability in _probabilities(counts) if probability > 0)


def normalized_entropy(counts: Sequence[int | float]) -> float:
    """Return entropy divided by the maximum entropy for the class count."""

    class_count = len(counts)
    return 0.0 if class_count <= 1 else entropy_bits(counts) / math.log2(class_count)


def jensen_shannon_divergence_bits(
    first_counts: Sequence[int | float], second_counts: Sequence[int | float]
) -> float:
    """Return zero-safe Jensen-Shannon divergence in bits."""

    if len(first_counts) != len(second_counts) or not first_counts:
        raise ValueError("Jensen-Shannon inputs must have the same nonzero length")
    first = _probabilities(first_counts)
    second = _probabilities(second_counts)
    midpoint = [(left + right) / 2 for left, right in zip(first, second, strict=True)]

    def divergence(values: Sequence[float]) -> float:
        return sum(
            value * math.log2(value / middle)
            for value, middle in zip(values, midpoint, strict=True)
            if value > 0 and middle > 0
        )

    result = (divergence(first) + divergence(second)) / 2
    if not math.isfinite(result):
        raise ValueError("Jensen-Shannon divergence is not finite")
    return max(0.0, result)


def population_statistics(values: Iterable[int | float]) -> dict[str, float]:
    """Return the required aggregate statistics for a nonempty sequence."""

    resolved = [float(value) for value in values]
    if not resolved or any(not math.isfinite(value) for value in resolved):
        raise ValueError("aggregate statistics require finite values")
    return {
        "minimum": min(resolved),
        "maximum": max(resolved),
        "mean": statistics.mean(resolved),
        "median": statistics.median(resolved),
        "population_standard_deviation": statistics.pstdev(resolved),
    }


def partition_diagnostics(
    client_indices: Sequence[Sequence[int]],
    labels: Sequence[int],
    eligible_indices: Sequence[int],
) -> tuple[list[dict], dict[str, dict[str, float]], dict[str, int]]:
    """Build per-client and aggregate label diagnostics."""

    eligible = [int(index) for index in eligible_indices]
    label_values = sorted({int(labels[index]) for index in eligible})
    if not label_values:
        raise ValueError("eligible training indices must contain at least one label")
    complete_counts = {
        str(label): sum(int(labels[index]) == label for index in eligible) for label in label_values
    }
    reference = [complete_counts[str(label)] for label in label_values]
    records: list[dict] = []
    for indices in client_indices:
        ordered = sorted(int(index) for index in indices)
        counts = {str(label): sum(int(labels[index]) == label for index in ordered) for label in label_values}
        values = [counts[str(label)] for label in label_values]
        represented = sum(value > 0 for value in values)
        records.append(
            {
                "class_counts": counts,
                "represented_labels": represented,
                "missing_labels": len(label_values) - represented,
                "label_entropy_bits": entropy_bits(values),
                "normalized_label_entropy": normalized_entropy(values),
                "jensen_shannon_divergence_bits": jensen_shannon_divergence_bits(values, reference),
                "example_count": len(ordered),
            }
        )
    measures = (
        "represented_labels",
        "missing_labels",
        "label_entropy_bits",
        "normalized_label_entropy",
        "jensen_shannon_divergence_bits",
        "example_count",
    )
    aggregate = {measure: population_statistics(record[measure] for record in records) for measure in measures}
    return records, aggregate, complete_counts


def diagnostics_from_artifact(
    artifact: Mapping,
    labels: Sequence[int],
    eligible_indices: Sequence[int],
) -> dict:
    """Derive current diagnostics from either historical or current partition artifacts."""

    clients = artifact.get("clients")
    if not isinstance(clients, list) or not clients:
        raise ValueError("partition artifact has no clients")
    records, aggregate, complete_counts = partition_diagnostics(
        [client.get("indices", []) for client in clients], labels, eligible_indices
    )
    return {
        "complete_eligible_training_class_counts": complete_counts,
        "clients": [
            {"client_id": client.get("client_id"), **record}
            for client, record in zip(clients, records, strict=True)
        ],
        "aggregate_statistics": aggregate,
    }
