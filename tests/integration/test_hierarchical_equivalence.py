import math
import warnings

import pytest
import torch

from fedapfa.federated.aggregation import (
    aggregation_accumulator_dtype,
    aggregation_tensor_policy,
    aggregation_weights,
    build_sufficient_statistics,
    combine_sufficient_statistics,
    sufficient_statistics_payload_bytes,
    weighted_fedavg,
)
from fedapfa.federated.numerical_equivalence import classify_model_states, prediction_identity
from fedapfa.federated.round_state import AggregationInput


def _input(client_id, examples, value, counter=3, enabled=True):
    return AggregationInput(
        client_id,
        examples,
        {
            "weight": torch.tensor([value, value + 1], dtype=torch.float32),
            "counter": torch.tensor(counter, dtype=torch.int64),
            "enabled": torch.tensor(enabled, dtype=torch.bool),
        },
    )


def _normalized_weight_mapping(updates, policy):
    return dict(
        zip(
            [item.client_id for item in updates],
            aggregation_weights(updates, policy),
            strict=True,
        )
    )


def _pre_week6_reference(inputs, policy):
    if policy == "uniform":
        weights = [1.0 / len(inputs) for _ in inputs]
    elif policy == "example_count":
        total = sum(item.example_count for item in inputs)
        weights = [item.example_count / total for item in inputs]
    else:
        raise ValueError(f"unsupported aggregation weighting: {policy}")
    reference = inputs[0].state_dict
    aggregated = {}
    for name, reference_value in reference.items():
        if reference_value.is_floating_point() or reference_value.is_complex():
            accumulator = torch.zeros_like(reference_value, dtype=torch.float64)
            for weight, item in zip(weights, inputs, strict=True):
                accumulator.add_(item.state_dict[name].detach().to(dtype=torch.float64), alpha=weight)
            aggregated[name] = accumulator.to(dtype=reference_value.dtype)
        else:
            aggregated[name] = reference_value.detach().clone()
    return aggregated, weights


def test_node_weighted_terms_and_global_combination_match_established_fedavg():
    updates = [_input("a", 2, 1), _input("b", 3, 2), _input("c", 5, 4), _input("d", 7, 8)]
    normalized_weights = _normalized_weight_mapping(updates, "example_count")
    node_zero = build_sufficient_statistics(
        updates[:2],
        policy="example_count",
        normalized_weights={item.client_id: normalized_weights[item.client_id] for item in updates[:2]},
    )
    node_one = build_sufficient_statistics(
        updates[2:],
        policy="example_count",
        normalized_weights={item.client_id: normalized_weights[item.client_id] for item in updates[2:]},
    )
    assert node_zero.weight_mass == pytest.approx(5 / 17)
    assert node_one.weight_mass == pytest.approx(12 / 17)
    expected_node_zero = torch.zeros(2, dtype=torch.float64)
    for item in updates[:2]:
        expected_node_zero.add_(
            item.state_dict["weight"].to(torch.float64),
            alpha=normalized_weights[item.client_id],
        )
    assert torch.equal(node_zero.weighted_sums["weight"], expected_node_zero)
    hierarchical, weights = combine_sufficient_statistics(
        [node_zero, node_one],
        policy="example_count",
        expected_client_ids=["a", "b", "c", "d"],
    )
    flat, flat_weights = weighted_fedavg(updates, "example_count")
    assert weights == flat_weights
    assert hierarchical.keys() == flat.keys()
    assert torch.allclose(hierarchical["weight"], flat["weight"], atol=1e-6, rtol=1e-6)
    assert hierarchical["counter"].item() == 3
    assert hierarchical["enabled"].item() is True
    assert sufficient_statistics_payload_bytes(node_one) > 0


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128],
)
@pytest.mark.parametrize("policy", ["example_count", "uniform"])
def test_flat_and_hierarchical_share_accumulation_casting_and_normalization(dtype, policy):
    updates = []
    for client_id, examples, value in (("a", 2, 1.25), ("b", 3, -0.5), ("c", 5, 4.0)):
        updates.append(
            AggregationInput(
                client_id,
                examples,
                {
                    "weight": torch.tensor([value, value + 0.25], dtype=dtype),
                    "counter": torch.tensor(3, dtype=torch.int64),
                    "enabled": torch.tensor(True, dtype=torch.bool),
                },
            )
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        one_contribution = build_sufficient_statistics(updates, policy=policy)
        one_group, one_weights = combine_sufficient_statistics(
            [one_contribution], policy=policy, expected_client_ids=["a", "b", "c"]
        )
        flat, flat_weights = weighted_fedavg(updates, policy)
    assert one_contribution.weighted_sums["weight"].dtype == aggregation_accumulator_dtype(dtype)
    assert one_weights == flat_weights
    assert all(torch.equal(one_group[name], flat[name]) for name in flat)

    normalized_weights = _normalized_weight_mapping(updates, policy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        grouped, grouped_weights = combine_sufficient_statistics(
            [
                build_sufficient_statistics(
                    updates[:2],
                    policy=policy,
                    normalized_weights={
                        item.client_id: normalized_weights[item.client_id]
                        for item in updates[:2]
                    },
                ),
                build_sufficient_statistics(
                    updates[2:],
                    policy=policy,
                    normalized_weights={
                        item.client_id: normalized_weights[item.client_id]
                        for item in updates[2:]
                    },
                ),
            ],
            policy=policy,
            expected_client_ids=["a", "b", "c"],
        )
    assert grouped_weights == flat_weights
    assert grouped["weight"].dtype == flat["weight"].dtype == dtype
    assert torch.allclose(grouped["weight"], flat["weight"], atol=1e-3, rtol=1e-3)
    assert torch.equal(grouped["counter"], flat["counter"])
    assert torch.equal(grouped["enabled"], flat["enabled"])
    assert aggregation_tensor_policy()["normalization"].startswith("normalize scalar client weights")


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128],
)
@pytest.mark.parametrize("policy", ["example_count", "uniform"])
def test_flat_weighted_fedavg_is_byte_identical_to_pre_week6_reference(dtype, policy):
    base_values = (1.25, -0.5, 4.0)
    values = tuple(complex(value, value / 2) for value in base_values) if dtype.is_complex else base_values
    updates = [
        AggregationInput(
            client_id,
            examples,
            {
                "weight": torch.tensor([value, value + 0.25], dtype=dtype),
                "counter": torch.tensor(3, dtype=torch.int64),
                "enabled": torch.tensor(True, dtype=torch.bool),
            },
        )
        for client_id, examples, value in zip(("a", "b", "c"), (2, 3, 5), values, strict=True)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expected, expected_weights = _pre_week6_reference(updates, policy)
        actual, actual_weights = weighted_fedavg(updates, policy)
    assert actual_weights == expected_weights
    assert actual.keys() == expected.keys()
    for name in expected:
        assert actual[name].dtype == expected[name].dtype
        assert torch.equal(
            actual[name].reshape(-1).view(torch.uint8),
            expected[name].reshape(-1).view(torch.uint8),
        )


def test_established_flat_uses_float64_and_normalizes_weights_before_accumulation():
    cancellation = [
        AggregationInput(str(index), 1, {"weight": torch.tensor([value], dtype=torch.float32)})
        for index, value in enumerate((1e8, 1.0, -1e8))
    ]
    established, weights = weighted_fedavg(cancellation, "uniform")
    float32_accumulator = torch.zeros(1, dtype=torch.float32)
    for weight, item in zip(weights, cancellation, strict=True):
        float32_accumulator.add_(item.state_dict["weight"], alpha=weight)
    assert not torch.equal(established["weight"], float32_accumulator)
    assert torch.equal(established["weight"], torch.tensor([1 / 3], dtype=torch.float32))

    normalization_order = [
        AggregationInput(str(index), examples, {"weight": torch.tensor([value], dtype=torch.float64)})
        for index, (examples, value) in enumerate(zip((1, 2, 3), (-1e16, -1e16, -1.0), strict=True))
    ]
    established, _ = weighted_fedavg(normalization_order, "example_count")
    divide_at_end = sum(
        item.state_dict["weight"] * item.example_count for item in normalization_order
    ) / sum(item.example_count for item in normalization_order)
    assert torch.equal(established["weight"], torch.tensor([-5e15], dtype=torch.float64))
    assert not torch.equal(established["weight"], divide_at_end)


def test_hierarchy_rejects_missing_duplicate_nonfinite_and_mismatched_buffers():
    first = build_sufficient_statistics([_input("a", 1, 1)], policy="example_count")
    duplicate = build_sufficient_statistics([_input("a", 1, 2)], policy="example_count")
    with pytest.raises(ValueError, match="duplicate client"):
        combine_sufficient_statistics([first, duplicate], expected_client_ids=["a"])
    with pytest.raises(ValueError, match="mismatch"):
        combine_sufficient_statistics([first], expected_client_ids=["a", "b"])
    with pytest.raises(ValueError, match="NaN|infinity"):
        build_sufficient_statistics([_input("a", 1, math.nan)])
    with pytest.raises(ValueError, match="non-floating"):
        build_sufficient_statistics([_input("a", 1, 1), _input("b", 1, 2, counter=4)])
    with pytest.raises(ValueError, match="non-floating"):
        weighted_fedavg([_input("a", 1, 1), _input("b", 1, 2, counter=4)])
    first.weighted_sums["weight"] = first.weighted_sums["weight"].to(torch.float32)
    with pytest.raises(ValueError, match="incompatible node weighted-sum"):
        combine_sufficient_statistics([first], expected_client_ids=["a"])


def test_numerical_classification_reports_bits_bounds_and_predictions():
    reference = {"x": torch.tensor([1.0, 2.0]), "count": torch.tensor(2)}
    exact = classify_model_states(
        reference,
        {name: value.clone() for name, value in reference.items()},
        absolute_tolerance=1e-6,
        relative_tolerance=1e-5,
    )
    assert exact.structural_identity and exact.mathematical_equivalence
    assert exact.bitwise_parameter_identity
    changed = {"x": torch.tensor([1.0 + 5e-7, 2.0]), "count": torch.tensor(2)}
    bounded = classify_model_states(reference, changed, absolute_tolerance=1e-6, relative_tolerance=1e-5)
    assert bounded.mathematical_equivalence and not bounded.bitwise_parameter_identity
    assert 0 < bounded.maximum_absolute_parameter_difference <= 1e-6
    assert bounded.maximum_relative_parameter_difference > 0
    assert prediction_identity([1, 0, 1], [1, 0, 1]) is True
    assert prediction_identity([1], [0]) is False
    assert prediction_identity(None, [0]) is None
