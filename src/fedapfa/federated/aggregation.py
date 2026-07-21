"""Validated configurable Federated Averaging."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from .round_state import AggregationInput

AGGREGATION_TENSOR_POLICY_VERSION = "established_flat_weighted_terms_v1"


def aggregation_accumulator_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the pre-Week-6 accumulator dtype used by established FedAvg."""

    if dtype.is_floating_point or dtype.is_complex:
        return torch.float64
    raise ValueError("non-floating tensors do not use a numeric accumulator")


def aggregation_tensor_policy() -> dict:
    return {
        "version": AGGREGATION_TENSOR_POLICY_VERSION,
        "real_floating_accumulator_dtype": "float64",
        "complex_input_accumulator_dtype": "float64",
        "complex_input_casting": (
            "cast to float64 before accumulation, discarding the imaginary component as in "
            "established FedAvg"
        ),
        "weighting": (
            "compute configured uniform or example_count normalized Python-float weights "
            "before tensor accumulation"
        ),
        "multiplication": "cast each input to float64, then ordered add_ with its normalized weight as alpha",
        "contribution_order": (
            "flat uses selected-client order; hierarchy preserves local selected order and "
            "groups only by node rank"
        ),
        "floating_casting": (
            "cast each input before weighted addition and cast the completed accumulator to "
            "the input dtype"
        ),
        "nonfloating_buffers": "require exact equality across every client and preserve the input dtype",
        "validation": (
            "identical keys shapes dtypes; positive example counts; finite inputs weights "
            "weighted terms and result"
        ),
        "normalization": (
            "normalize scalar client weights before ordered accumulation; do not divide the "
            "completed tensor sum"
        ),
    }


@dataclass(frozen=True)
class SufficientStatistics:
    """Mergeable node-grouped terms using established normalized weights."""

    policy: str
    client_ids: tuple[str, ...]
    example_counts: tuple[int, ...]
    normalized_weights: tuple[float, ...]
    weight_mass: float
    weighted_sums: dict[str, torch.Tensor]
    nonfloating_state: dict[str, torch.Tensor]
    output_dtypes: dict[str, torch.dtype]
    output_shapes: dict[str, tuple[int, ...]]


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


def build_sufficient_statistics(
    inputs: Sequence[AggregationInput],
    *,
    policy: str = "example_count",
    normalized_weights: Mapping[str, float] | None = None,
) -> SufficientStatistics:
    """Build one node's established weighted terms without changing their arithmetic."""

    items = list(inputs)
    weights = aggregation_weights(items, policy)
    if len({item.client_id for item in items}) != len(items):
        raise ValueError("duplicate client ids in sufficient-statistics input")
    if normalized_weights is not None:
        if set(normalized_weights) != {str(item.client_id) for item in items}:
            raise ValueError("normalized client weights do not match the node clients")
        weights = [normalized_weights[str(item.client_id)] for item in items]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in weights
        ):
            raise ValueError("normalized client weights must be finite and nonnegative")
        weights = [float(value) for value in weights]
    weight_mass = sum(weights)
    if not math.isfinite(weight_mass) or weight_mass <= 0 or weight_mass > 1.0 + 1e-12:
        raise ValueError("node normalized-weight mass must be finite and in (0, 1]")

    reference = items[0].state_dict
    if not reference:
        raise ValueError("client state dictionary is empty")

    for item in items:
        if set(item.state_dict) != set(reference):
            raise ValueError(f"client {item.client_id} has incompatible state keys")
        for name, value in item.state_dict.items():
            expected = reference[name]
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"client {item.client_id} has incompatible tensor {name}")
            if not _finite_tensor(value):
                raise ValueError(f"client {item.client_id} tensor {name} contains NaN or infinity")

    weighted_sums: dict[str, torch.Tensor] = {}
    nonfloating_state: dict[str, torch.Tensor] = {}
    output_dtypes: dict[str, torch.dtype] = {}
    output_shapes: dict[str, tuple[int, ...]] = {}
    for name, reference_value in reference.items():
        reference_tensor = reference_value.detach()
        output_dtypes[name] = reference_tensor.dtype
        output_shapes[name] = tuple(reference_tensor.shape)
        if reference_tensor.is_floating_point() or reference_tensor.is_complex():
            weighted_sum = torch.zeros_like(reference_tensor, dtype=torch.float64)
            for weight, item in zip(weights, items, strict=True):
                weighted_sum.add_(
                    item.state_dict[name].detach().to(dtype=torch.float64),
                    alpha=weight,
                )
            if not torch.isfinite(weighted_sum).all():
                raise ValueError(f"node weighted-sum tensor {name} contains NaN or infinity")
            weighted_sums[name] = weighted_sum
        else:
            if any(not torch.equal(item.state_dict[name], reference_value) for item in items[1:]):
                raise ValueError(f"non-floating state tensor {name} differs across clients")
            nonfloating_state[name] = reference_tensor.clone()

    return SufficientStatistics(
        policy=policy,
        client_ids=tuple(str(item.client_id) for item in items),
        example_counts=tuple(int(item.example_count) for item in items),
        normalized_weights=tuple(weights),
        weight_mass=weight_mass,
        weighted_sums=weighted_sums,
        nonfloating_state=nonfloating_state,
        output_dtypes=output_dtypes,
        output_shapes=output_shapes,
    )


def combine_sufficient_statistics(
    contributions: Sequence[SufficientStatistics],
    *,
    policy: str = "example_count",
    expected_client_ids: Sequence[int] | None = None,
) -> tuple[dict[str, torch.Tensor], list[float]]:
    """Combine established weighted terms in node-rank order without renormalizing."""

    if policy not in {"example_count", "uniform"}:
        raise ValueError(f"unsupported aggregation weighting: {policy}")
    node_statistics = list(contributions)
    if not node_statistics:
        raise ValueError("cannot combine an empty sufficient-statistics collection")
    first = node_statistics[0]
    expected_names = set(first.output_dtypes)
    client_example_counts: dict[str, int] = {}
    client_normalized_weights: dict[str, float] = {}
    observed_client_ids: list[str] = []
    weight_mass = 0.0

    for statistics in node_statistics:
        if statistics.policy != policy:
            raise ValueError("sufficient-statistics weighting policies differ")
        if set(statistics.output_dtypes) != expected_names:
            raise ValueError("sufficient-statistics state keys differ")
        if statistics.output_dtypes != first.output_dtypes:
            raise ValueError("sufficient-statistics output dtypes differ")
        if statistics.output_shapes != first.output_shapes:
            raise ValueError("sufficient-statistics output shapes differ")
        if not (
            len(statistics.client_ids)
            == len(statistics.example_counts)
            == len(statistics.normalized_weights)
        ):
            raise ValueError("client ids, example counts, and normalized weights have different lengths")
        if (
            not math.isfinite(statistics.weight_mass)
            or statistics.weight_mass <= 0
            or not math.isclose(
                statistics.weight_mass,
                sum(statistics.normalized_weights),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("node normalized-weight mass is incompatible")
        weight_mass += statistics.weight_mass
        for client_id, example_count, normalized_weight in zip(
            statistics.client_ids,
            statistics.example_counts,
            statistics.normalized_weights,
            strict=True,
        ):
            if client_id in client_example_counts:
                raise ValueError(f"duplicate client id across nodes: {client_id}")
            if (
                example_count <= 0
                or not math.isfinite(normalized_weight)
                or normalized_weight < 0
            ):
                raise ValueError("client example counts or normalized weights are incompatible")
            observed_client_ids.append(client_id)
            client_example_counts[client_id] = example_count
            client_normalized_weights[client_id] = normalized_weight

    ordered_client_ids = (
        observed_client_ids if expected_client_ids is None else [str(client_id) for client_id in expected_client_ids]
    )
    if len(set(ordered_client_ids)) != len(ordered_client_ids):
        raise ValueError("expected client ids contain duplicates")
    if set(ordered_client_ids) != set(observed_client_ids):
        missing = sorted(set(ordered_client_ids) - set(observed_client_ids))
        extra = sorted(set(observed_client_ids) - set(ordered_client_ids))
        raise ValueError(f"sufficient-statistics clients mismatch: missing={missing}, extra={extra}")
    expected_weight_inputs = [
        AggregationInput(client_id, client_example_counts[client_id], {})
        for client_id in ordered_client_ids
    ]
    expected_weights = aggregation_weights(expected_weight_inputs, policy)
    if any(
        client_normalized_weights[client_id] != expected_weight
        for client_id, expected_weight in zip(ordered_client_ids, expected_weights, strict=True)
    ):
        raise ValueError("node normalized weights differ from established FedAvg weights")
    if not math.isclose(weight_mass, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("combined normalized-weight mass does not sum to one")

    aggregated: dict[str, torch.Tensor] = {}
    for name, output_dtype in first.output_dtypes.items():
        if output_dtype.is_floating_point or output_dtype.is_complex:
            accumulator_dtype = aggregation_accumulator_dtype(output_dtype)
            first_weighted_sum = first.weighted_sums.get(name)
            if first_weighted_sum is None:
                raise ValueError(f"missing node weighted-sum tensor {name}")
            combined = torch.zeros_like(first_weighted_sum, dtype=accumulator_dtype)
            for statistics in node_statistics:
                weighted_sum = statistics.weighted_sums.get(name)
                if weighted_sum is None:
                    raise ValueError(f"missing node weighted-sum tensor {name}")
                if (
                    weighted_sum.dtype != accumulator_dtype
                    or tuple(weighted_sum.shape) != first.output_shapes[name]
                    or not torch.isfinite(weighted_sum).all()
                ):
                    raise ValueError(f"incompatible node weighted-sum tensor {name}")
                combined.add_(weighted_sum)
            value = combined.to(dtype=output_dtype)
            if not torch.isfinite(value).all():
                raise ValueError(f"aggregated tensor {name} contains NaN or infinity")
            aggregated[name] = value
        else:
            values = [statistics.nonfloating_state.get(name) for statistics in node_statistics]
            if any(value is None for value in values):
                raise ValueError(f"missing non-floating state tensor {name}")
            reference_value = values[0]
            assert reference_value is not None
            if any(not torch.equal(reference_value, value) for value in values[1:] if value is not None):
                raise ValueError(f"non-floating state tensor {name} differs across nodes")
            aggregated[name] = reference_value.clone()

    return aggregated, expected_weights


def sufficient_statistics_payload_bytes(statistics: SufficientStatistics) -> int:
    """Return logical tensor payload bytes for a node contribution."""

    tensors = (*statistics.weighted_sums.values(), *statistics.nonfloating_state.values())
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


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
