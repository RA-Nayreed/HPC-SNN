import math

import torch

from fedapfa.neurons.surrogate import atan_spike


def test_surrogate_forward_and_exact_backward():
    value = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    output = atan_spike(value, 5.0)
    assert output.tolist() == [0.0, 1.0, 1.0]
    output.sum().backward()
    expected = 5 / (2 * (1 + (math.pi * 5 * value.detach() / 2).square()))
    assert torch.allclose(value.grad, expected)
