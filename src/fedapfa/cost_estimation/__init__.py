"""Deterministic client-cost data, regression, and offline assignment evaluation."""

from .dataset import build_client_cost_dataset, validate_accepted_run
from .decision import decide_spike_history
from .regression import CostModel, fit_regression

__all__ = [
    "CostModel",
    "build_client_cost_dataset",
    "decide_spike_history",
    "fit_regression",
    "validate_accepted_run",
]
