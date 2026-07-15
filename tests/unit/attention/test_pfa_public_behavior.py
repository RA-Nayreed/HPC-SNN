import pytest
import torch

from fedapfa.attention import EquationPFA, PublicBehaviorPFA


def test_public_behavior_matches_pinned_golden_equation():
    current = torch.tensor([[[0.0, 1.0, 10.0]]])
    mean = current.mean(-1, keepdim=True)
    squared = (current - mean).square()
    weight = (current - mean) / (3 * squared + 0.02)
    bias = (1 - weight * (current + mean)) / 2
    expected = current * torch.sigmoid(weight * current + bias)
    actual = PublicBehaviorPFA(0.01)(current)
    assert torch.allclose(actual, expected)
    assert not torch.allclose(actual, EquationPFA(0.01)(current))
    assert not list(PublicBehaviorPFA().parameters())


@pytest.mark.parametrize(
    "current",
    [
        torch.ones(1, 2, 4),
        torch.tensor([[[0.0, 0.0, 1.0, 0.0]]]),
        torch.tensor([[[0.2, -0.3, 0.7, 1.1], [2.0, 0.5, -1.0, 0.0]]]),
        torch.tensor([[[0.0, 1.0, 100.0, -3.0]]]),
    ],
)
def test_both_pfa_variants_are_finite_and_differentiable(current):
    for module in (EquationPFA(0.01), PublicBehaviorPFA(0.01)):
        value = current.clone().requires_grad_(True)
        output = module(value)
        output.sum().backward()
        assert output.shape == value.shape
        assert torch.isfinite(output).all()
        assert torch.isfinite(value.grad).all()
        assert not list(module.parameters())
