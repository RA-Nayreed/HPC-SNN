"""Server-side FedAvg application for typed client results."""

from __future__ import annotations

from torch import nn

from .aggregation import (
    clone_state_dict,
    state_difference_cosine_similarity,
    state_difference_l2_norm,
    weighted_fedavg,
)
from .round_state import AggregationInput, ClientResult


def aggregate_client_results(
    model: nn.Module, results: list[ClientResult]
) -> tuple[list[float], float, list[float]]:
    """Aggregate client states and calculate alignment while updates are resident."""

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
    cosines = [
        state_difference_cosine_similarity(result.state_dict, before, aggregated) for result in results
    ]
    update_norm = state_difference_l2_norm(aggregated, before)
    model.load_state_dict(aggregated, strict=True)
    return weights, update_norm, cosines
