"""Validated sample-count-weighted Federated Averaging."""

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


def state_difference_l2_norm(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> float:
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


def aggregation_weights(inputs: list[AggregationInput]) -> list[float]:
    if not inputs:
        raise ValueError("cannot aggregate an empty client-update collection")
    if any(not isinstance(item.example_count, int) or item.example_count <= 0 for item in inputs):
        raise ValueError("client example counts must be positive integers")
    total = sum(item.example_count for item in inputs)
    weights = [item.example_count / total for item in inputs]
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("aggregation weights must be finite and nonnegative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("aggregation weights do not sum to one")
    return weights


def weighted_fedavg(inputs: list[AggregationInput]) -> tuple[dict[str, torch.Tensor], list[float]]:
    weights = aggregation_weights(inputs)
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
