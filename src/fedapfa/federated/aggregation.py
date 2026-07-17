"""Validated configurable Federated Averaging."""

from __future__ import annotations

import math

import torch

from .round_state import AggregationInput


def _finite_tensor(value: torch.Tensor) -> bool:
    return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all())


def clone_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in state.items()}


def state_l2_norm(state: dict[str, torch.Tensor]) -> float:
    squared = 0.0
    for value in state.values():
        if value.is_floating_point() or value.is_complex():
            if not _finite_tensor(value):
                raise ValueError("model state contains NaN or infinity")
            squared += float(torch.sum(value.detach().to(dtype=torch.float64).abs().square()))
    result = math.sqrt(squared)
    if not math.isfinite(result):
        raise ValueError("model norm is not finite")
    return result


def state_difference_l2_norm(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> float:
    if set(first) != set(second):
        raise ValueError("state dictionaries have incompatible keys")
    squared = 0.0
    for name, value in first.items():
        other = second[name]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError(f"state tensor {name} is incompatible")
        if value.is_floating_point() or value.is_complex():
            difference = value.detach().to(dtype=torch.float64) - other.detach().to(dtype=torch.float64)
            if not torch.isfinite(difference).all():
                raise ValueError("model update contains NaN or infinity")
            squared += float(torch.sum(difference.abs().square()))
        elif not torch.equal(value, other):
            raise ValueError(f"non-floating state tensor {name} changed across client updates")
    result = math.sqrt(squared)
    if not math.isfinite(result):
        raise ValueError("model update norm is not finite")
    return result


def state_difference_cosine_similarity(
    client_state: dict[str, torch.Tensor],
    server_state: dict[str, torch.Tensor],
    aggregated_state: dict[str, torch.Tensor],
) -> float:
    """Compare a client update with the aggregated update; zero norms map to zero."""

    if set(client_state) != set(server_state) or set(client_state) != set(aggregated_state):
        raise ValueError("state dictionaries have incompatible keys")
    dot_product = client_squared = aggregate_squared = 0.0
    for name, client_value in client_state.items():
        server_value = server_state[name]
        aggregate_value = aggregated_state[name]
        if client_value.shape != server_value.shape or client_value.shape != aggregate_value.shape:
            raise ValueError(f"state tensor {name} is incompatible")
        if client_value.is_floating_point() or client_value.is_complex():
            client_update = client_value.detach().to(dtype=torch.float64) - server_value.detach().to(
                dtype=torch.float64
            )
            aggregate_update = aggregate_value.detach().to(dtype=torch.float64) - server_value.detach().to(
                dtype=torch.float64
            )
            if not torch.isfinite(client_update).all() or not torch.isfinite(aggregate_update).all():
                raise ValueError("model update contains NaN or infinity")
            dot_product += float(torch.sum(client_update * aggregate_update))
            client_squared += float(torch.sum(client_update.abs().square()))
            aggregate_squared += float(torch.sum(aggregate_update.abs().square()))
    if client_squared == 0 or aggregate_squared == 0:
        return 0.0
    result = dot_product / math.sqrt(client_squared * aggregate_squared)
    if not math.isfinite(result):
        raise ValueError("model-update cosine similarity is not finite")
    return min(1.0, max(-1.0, result))


def aggregation_weights(inputs: list[AggregationInput], policy: str = "example_count") -> list[float]:
    if not inputs:
        raise ValueError("cannot aggregate an empty client-update collection")
    if any(not isinstance(item.example_count, int) or item.example_count <= 0 for item in inputs):
        raise ValueError("client example counts must be positive integers")
    if policy == "uniform":
        weight = 1.0 / len(inputs)
        weights = [weight for _ in inputs]
    elif policy == "example_count":
        total = sum(item.example_count for item in inputs)
        weights = [item.example_count / total for item in inputs]
    else:
        raise ValueError(f"unsupported aggregation weighting: {policy}")
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("aggregation weights must be finite and nonnegative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("aggregation weights do not sum to one")
    return weights


def weighted_fedavg(
    inputs: list[AggregationInput], policy: str = "example_count"
) -> tuple[dict[str, torch.Tensor], list[float]]:
    weights = aggregation_weights(inputs, policy)
    reference = inputs[0].state_dict
    if not reference:
        raise ValueError("client state dictionary is empty")
    for item in inputs:
        if set(item.state_dict) != set(reference):
            raise ValueError(f"client {item.client_id} has incompatible state keys")
        for name, value in item.state_dict.items():
            expected = reference[name]
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"client {item.client_id} has incompatible tensor {name}")
            if not _finite_tensor(value):
                raise ValueError(f"client {item.client_id} tensor {name} contains NaN or infinity")

    aggregated: dict[str, torch.Tensor] = {}
    for name, reference_value in reference.items():
        if reference_value.is_floating_point() or reference_value.is_complex():
            accumulator = torch.zeros_like(reference_value, dtype=torch.float64)
            for weight, item in zip(weights, inputs, strict=True):
                accumulator.add_(item.state_dict[name].detach().to(dtype=torch.float64), alpha=weight)
            value = accumulator.to(dtype=reference_value.dtype)
            if not torch.isfinite(value).all():
                raise ValueError(f"aggregated tensor {name} contains NaN or infinity")
            aggregated[name] = value
        else:
            if any(not torch.equal(item.state_dict[name], reference_value) for item in inputs[1:]):
                raise ValueError(f"non-floating state tensor {name} differs across clients")
            aggregated[name] = reference_value.detach().clone()
    return aggregated, weights
