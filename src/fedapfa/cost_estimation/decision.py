"""Predeclared adoption rule for causal spike-history features."""

from __future__ import annotations


def _conditions(
    history: dict, reference: dict, rank_tolerance: float, minimum_improvement: float
) -> dict:
    history_median = history["median_absolute_error"]
    reference_median = reference["median_absolute_error"]
    improvement = (
        (reference_median - history_median) / reference_median
        if reference_median > 0
        else 0.0
    )
    return {
        "median_absolute_runtime_error_improvement_fraction": improvement,
        "median_error_improves_by_declared_fraction": improvement >= minimum_improvement,
        "p90_absolute_runtime_error_does_not_worsen": (
            history["p90_absolute_error"] <= reference["p90_absolute_error"]
        ),
        "rank_correlation_is_maintained": (
            history["spearman_rank_correlation"]
            >= reference["spearman_rank_correlation"] - rank_tolerance
        ),
    }


def decide_spike_history(
    per_dataset_metrics: dict[str, dict[str, dict]],
    prediction_time_fraction: float,
    assignment_regret: dict[str, float],
    rank_tolerance: float = 0.01,
    prediction_time_limit: float = 0.001,
    minimum_improvement: float = 0.05,
) -> dict:
    """Adopt only when every untouched seed-27 condition holds on SHD and SSC."""

    conditions = {}
    for dataset in ("shd", "ssc"):
        if dataset not in per_dataset_metrics:
            raise ValueError("spike decision requires SHD and SSC metrics")
        conditions[dataset] = _conditions(
            per_dataset_metrics[dataset]["historical_spike"],
            per_dataset_metrics[dataset]["strongest_non_spike"],
            rank_tolerance,
            minimum_improvement,
        )
    timing = prediction_time_fraction <= prediction_time_limit
    assignment = assignment_regret["historical_spike"] < assignment_regret["strongest_non_spike"]
    adopted = all(
        all(value for value in result.values() if isinstance(value, bool))
        for result in conditions.values()
    )
    adopted = adopted and timing and assignment
    return {
        "decision": "spike_history_adopted" if adopted else "spike_history_not_adopted",
        "dataset_conditions": conditions,
        "prediction_time_negligible": timing,
        "offline_assignment_closer_to_oracle": assignment,
        "prediction_time_fraction": prediction_time_fraction,
        "rank_correlation_tolerance": rank_tolerance,
        "minimum_runtime_error_improvement_fraction": minimum_improvement,
        "selected_scheduler_model": "historical_spike" if adopted else "strongest_non_spike",
    }


def ensure_exportable(model_name: str) -> None:
    if model_name == "diagnostic_oracle":
        raise ValueError("diagnostic oracle is unavailable before assignment and cannot be exported")
