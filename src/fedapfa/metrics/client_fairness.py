"""Distribution-weighted client validation fairness proxy."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence

PROXY_NAME = "client_distribution_weighted_validation_accuracy"
PROXY_EXPLANATION = (
    "This is a distribution-weighted proxy, not observed accuracy on private client test data."
)


def _percentile(values: Sequence[float], percentage: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def client_distribution_weighted_validation_accuracy(
    validation_per_class_accuracy: Sequence[float],
    client_class_counts: Mapping[str, int],
) -> float:
    """Weight global validation class accuracies by one client's train labels."""

    class_accuracies = [float(value) for value in validation_per_class_accuracy]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in class_accuracies):
        raise ValueError("validation per-class accuracies must be finite values in [0, 1]")
    counts = [int(client_class_counts.get(str(index), 0)) for index in range(len(class_accuracies))]
    if any(value < 0 for value in counts) or sum(counts) <= 0:
        raise ValueError("client class counts must have a positive total")
    return sum(accuracy * count for accuracy, count in zip(class_accuracies, counts, strict=True)) / sum(counts)


def summarize_client_fairness_proxy(values: Sequence[float]) -> dict[str, float]:
    """Summarize the client distribution-weighted proxy."""

    resolved = [float(value) for value in values]
    if not resolved or any(not math.isfinite(value) or not 0 <= value <= 1 for value in resolved):
        raise ValueError("client fairness proxy values must be finite values in [0, 1]")
    return {
        "minimum": min(resolved),
        "10th_percentile": _percentile(resolved, 0.1),
        "median": statistics.median(resolved),
        "mean": statistics.mean(resolved),
        "maximum": max(resolved),
        "population_standard_deviation": statistics.pstdev(resolved),
    }


def fairness_proxy_record(validation_per_class_accuracy: Sequence[float], partition_artifact: Mapping) -> dict:
    """Build per-client values and summary with an explicit limitation."""

    values = [
        {
            "client_id": client["client_id"],
            PROXY_NAME: client_distribution_weighted_validation_accuracy(
                validation_per_class_accuracy, client["class_counts"]
            ),
        }
        for client in partition_artifact["clients"]
    ]
    return {
        "name": PROXY_NAME,
        "definition": PROXY_EXPLANATION,
        "values": values,
        "statistics": summarize_client_fairness_proxy([record[PROXY_NAME] for record in values]),
    }
