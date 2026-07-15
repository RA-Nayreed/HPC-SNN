"""Server-side FedAvg application for typed client results."""

from __future__ import annotations

from torch import nn

from .aggregation import clone_state_dict, state_difference_l2_norm, weighted_fedavg
from .round_state import AggregationInput, ClientResult


def aggregate_client_results(model: nn.Module, results: list[ClientResult]) -> tuple[list[float], float]:
    before = clone_state_dict(model.state_dict())
    aggregated, weights = weighted_fedavg(
        [
            AggregationInput(
                client_id=result.client_id,
                example_count=result.example_count,
                state_dict=result.state_dict,
            )
            for result in results
        ]
    )
    model.load_state_dict(aggregated, strict=True)
    return weights, state_difference_l2_norm(aggregated, before)
