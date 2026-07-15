import pytest
import torch

from fedapfa.attention import EquationPFA


def test_equation_matches_independent_calculation():
    current = torch.tensor([[[1.0, 3.0, 7.0]]])
    mean = current.sum(-1, keepdim=True) / 3
    deviation = current - mean
    variance = (deviation * deviation).sum(-1, keepdim=True) / 3
    weight = deviation / (deviation * deviation + 0.02 + 2 * variance)
    bias = (1 - weight * (current + mean)) / 2
    expected = current / (1 + torch.exp(-(weight * current + bias)))
    assert torch.allclose(EquationPFA(0.01)(current), expected)


def test_parameter_free_shape_dtype_constant_gradient_and_grid():
    for value in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        module = EquationPFA(value)
        current = torch.ones(1, 2, 4, dtype=torch.float32, requires_grad=True)
        output = module(current)
        output.sum().backward()
        assert (
            output.shape == current.shape
            and output.dtype == current.dtype
            and torch.isfinite(output).all()
            and torch.isfinite(current.grad).all()
            and not list(module.parameters())
        )
    with pytest.raises(ValueError):
        EquationPFA(0)
