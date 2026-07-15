import math

import torch

from fedapfa.neurons.lif import EulerLIF, SpikingJellyLIF


def test_euler_membrane_threshold_and_subtractive_reset():
    neuron = EulerLIF(tau_ms=20, dt_ms=10, threshold=1)
    spikes = neuron(torch.tensor([[[0.5], [0.8]]]))
    assert spikes.squeeze().tolist() == [0.0, 1.0]
    assert torch.allclose(neuron.membrane, torch.tensor([[0.05]]))


def test_spikingjelly_charge_recurrence():
    neuron = SpikingJellyLIF(tau_ms=20, dt_ms=10, threshold=10)
    neuron(torch.tensor([[[2.0], [0.0]]]))
    assert torch.allclose(neuron.membrane, torch.tensor([[0.5]]))


def test_padding_is_ignored_and_state_resets_between_batches():
    neuron = EulerLIF(tau_ms=20, dt_ms=10)
    current = torch.tensor([[[1.2], [100.0]]])
    mask = torch.tensor([[True, False]])
    spikes = neuron(current, mask)
    assert spikes[0, 1, 0] == 0
    first = neuron(torch.tensor([[[0.6]]])).clone()
    second = neuron(torch.tensor([[[0.6]]])).clone()
    assert torch.equal(first, second)


def test_detached_reset_has_the_independent_expected_gradient():
    current = torch.tensor([[[1.1], [0.2]]], requires_grad=True)
    EulerLIF(tau_ms=20, dt_ms=10)(current).sum().backward()
    derivative0 = 5 / (2 * (1 + (math.pi * 5 * 0.1 / 2) ** 2))
    charged1 = 0.5 * 0.1 + 0.2
    derivative1 = 5 / (2 * (1 + (math.pi * 5 * (charged1 - 1) / 2) ** 2))
    assert torch.allclose(current.grad[0, 0, 0], torch.tensor(derivative0 + 0.5 * derivative1))


def test_zero_reset_clears_membrane_after_spike():
    neuron = EulerLIF(tau_ms=20, dt_ms=10, reset="zero")
    neuron(torch.tensor([[[1.2]]]))
    assert torch.equal(neuron.membrane, torch.zeros(1, 1))
