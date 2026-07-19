"""Cost prediction metrics and deterministic subgroup reports."""

from __future__ import annotations

import numpy as np


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    offset = 0
    while offset < len(values):
        end = offset + 1
        while end < len(values) and values[order[end]] == values[order[offset]]:
            end += 1
        ranks[order[offset:end]] = (offset + end - 1) / 2.0 + 1.0
        offset = end
    return ranks


def spearman_rank_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2:
        return 0.0
    left = _ranks(actual)
    right = _ranks(predicted)
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def regression_metrics(actual, predicted, percentage_denominator_floor: float) -> dict:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape or actual.ndim != 1 or len(actual) == 0:
        raise ValueError("metric inputs must be non-empty matching vectors")
    if percentage_denominator_floor <= 0:
        raise ValueError("percentage denominator floor must be positive")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("metric inputs must be finite")
    residual = predicted - actual
    absolute = np.abs(residual)
    percentage = absolute / np.maximum(np.abs(actual), percentage_denominator_floor)
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "mean_absolute_error": float(np.mean(absolute)),
        "median_absolute_error": float(np.median(absolute)),
        "p90_absolute_error": float(np.quantile(absolute, 0.9)),
        "root_mean_squared_error": float(np.sqrt(np.mean(residual**2))),
        "median_absolute_percentage_error": float(np.median(percentage)),
        "p90_absolute_percentage_error": float(np.quantile(percentage, 0.9)),
        "r_squared": 0.0 if denominator == 0 else float(1 - np.sum(residual**2) / denominator),
        "spearman_rank_correlation": spearman_rank_correlation(actual, predicted),
        "mean_signed_error": float(np.mean(residual)),
        "median_signed_error": float(np.median(residual)),
        "sample_count": len(actual),
        "percentage_denominator_floor": percentage_denominator_floor,
    }


def _quartile(values: np.ndarray) -> np.ndarray:
    boundaries = np.quantile(values, [0.25, 0.5, 0.75])
    return np.searchsorted(boundaries, values, side="right") + 1


def _subset_metrics(rows, actual, predicted, indexes, floor) -> dict:
    if not indexes:
        return {"sample_count": 0}
    selection = np.asarray(indexes, dtype=np.int64)
    return regression_metrics(actual[selection], predicted[selection], floor)


def evaluate_predictions(rows: list[dict], target: str, predicted, percentage_denominator_floor: float) -> dict:
    actual = np.asarray([float(row[target]) for row in rows], dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if len(rows) != len(predicted):
        raise ValueError("prediction count differs from evaluation row count")
    report = {"joint": regression_metrics(actual, predicted, percentage_denominator_floor)}
    for dimension, values in (
        ("dataset", sorted({str(row["dataset"]) for row in rows})),
        ("seed", sorted({int(row["scientific_seed"]) for row in rows})),
    ):
        report[f"per_{dimension}"] = {}
        field = "dataset" if dimension == "dataset" else "scientific_seed"
        for value in values:
            indexes = [index for index, row in enumerate(rows) if row[field] == value]
            report[f"per_{dimension}"][str(value)] = _subset_metrics(
                rows, actual, predicted, indexes, percentage_denominator_floor
            )
    for name, field in (
        ("client_size_quartile", "example_count"),
        ("sequence_length_quartile", "mean_sequence_length"),
    ):
        quartiles = _quartile(np.asarray([float(row[field]) for row in rows]))
        report[f"by_{name}"] = {
            str(quartile): _subset_metrics(
                rows,
                actual,
                predicted,
                [index for index, value in enumerate(quartiles) if value == quartile],
                percentage_denominator_floor,
            )
            for quartile in range(1, 5)
        }
    report["by_communication_round_interval"] = {}
    for lower, upper in ((1, 20), (21, 40), (41, 60), (61, 80), (81, 100)):
        indexes = [
            index
            for index, row in enumerate(rows)
            if lower <= int(row["communication_round"]) <= upper
        ]
        report["by_communication_round_interval"][f"{lower}-{upper}"] = _subset_metrics(
            rows, actual, predicted, indexes, percentage_denominator_floor
        )
    for name, expected in (("without_history", False), ("with_history", True)):
        indexes = [
            index for index, row in enumerate(rows) if bool(row["has_historical_observations"]) is expected
        ]
        report[name] = _subset_metrics(rows, actual, predicted, indexes, percentage_denominator_floor)
    return report
