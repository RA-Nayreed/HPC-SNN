import pytest
import torch

from fedapfa.federated.aggregation import aggregation_weights, weighted_fedavg
from fedapfa.federated.round_state import AggregationInput


def _input(client_id, examples, values):
    return AggregationInput(client_id, examples, {"weight": torch.tensor(values, dtype=torch.float32)})


def test_unequal_client_sizes_produce_exact_weighted_average():
    inputs = [_input("a", 1, [1.0, 3.0]), _input("b", 3, [5.0, 7.0])]
    state, weights = weighted_fedavg(inputs)
    assert weights == pytest.approx([0.25, 0.75])
    assert sum(weights) == pytest.approx(1.0)
    assert torch.equal(state["weight"], torch.tensor([4.0, 6.0]))


def test_uniform_average_uses_exact_equal_weights_and_aggregates_running_statistics():
    inputs = [
        AggregationInput(
            "a",
            1,
            {
                "weight": torch.tensor([1.0, 3.0]),
                "bntt.running_mean": torch.tensor([2.0]),
                "bntt.running_variance": torch.tensor([4.0]),
            },
        ),
        AggregationInput(
            "b",
            9,
            {
                "weight": torch.tensor([5.0, 7.0]),
                "bntt.running_mean": torch.tensor([6.0]),
                "bntt.running_variance": torch.tensor([8.0]),
            },
        ),
    ]
    state, weights = weighted_fedavg(inputs, "uniform")
    assert weights == [0.5, 0.5]
    assert torch.equal(state["weight"], torch.tensor([3.0, 5.0]))
    assert torch.equal(state["bntt.running_mean"], torch.tensor([4.0]))
    assert torch.equal(state["bntt.running_variance"], torch.tensor([6.0]))


def test_one_client_average_equals_its_local_result():
    item = _input("only", 9, [0.125, -2.5])
    state, weights = weighted_fedavg([item])
    assert weights == [1.0]
    assert torch.equal(state["weight"], item.state_dict["weight"])


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_updates_fail_explicitly(value):
    with pytest.raises(ValueError, match="NaN or infinity"):
        weighted_fedavg([_input("invalid", 1, [value])])


def test_empty_or_incompatible_updates_are_rejected():
    with pytest.raises(ValueError, match="empty"):
        aggregation_weights([])
    first = _input("a", 1, [1.0])
    second = AggregationInput("b", 1, {"other": torch.tensor([1.0])})
    with pytest.raises(ValueError, match="incompatible"):
        weighted_fedavg([first, second])
    with pytest.raises(ValueError, match="unsupported aggregation"):
        weighted_fedavg([first], "unknown")
