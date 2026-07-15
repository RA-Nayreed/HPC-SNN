import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from fedapfa.datasets.sequence_collation import collate_event_sequences
from fedapfa.metrics.spike_rate import spike_rate
from fedapfa.training.centralized import run_epoch


def test_spike_rate_excludes_padding_and_counts_neurons():
    spikes = torch.tensor([[[1.0, 0.0], [1.0, 1.0]], [[1.0, 1.0], [9.0, 9.0]]])
    mask = torch.tensor([[True, True], [True, False]])
    assert float(spike_rate(spikes, mask)) == pytest.approx(5 / 6)


class _VariableRateModel(nn.Module):
    def forward(self, inputs, lengths):
        valid = torch.arange(inputs.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        rates = {"layer1": spike_rate(inputs[..., :1], valid)}
        return torch.zeros((len(inputs), 2)), rates


def test_epoch_spike_rate_is_weighted_by_valid_timesteps():
    samples = [
        (torch.ones(1, 1), 0),
        (torch.ones(1, 1), 0),
        (torch.zeros(10, 1), 0),
    ]
    loader = DataLoader(samples, batch_size=2, collate_fn=collate_event_sequences)
    metrics = run_epoch(_VariableRateModel(), loader, torch.device("cpu"))
    assert metrics["spike_rates"]["layer1"] == pytest.approx(2 / 12)
