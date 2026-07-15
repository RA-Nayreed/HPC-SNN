"""Configuration-driven feed-forward event-audio LIF network."""

from __future__ import annotations

import torch
from torch import nn

from fedapfa.attention import make_attention
from fedapfa.metrics.spike_rate import spike_rate
from fedapfa.neurons.lif import make_lif


class AudioLIFSNN(nn.Module):
    def __init__(self, input_features, hidden_dims, classes, neuron, attention, dropout, batch_normalization, bias):
        super().__init__()
        first, second = hidden_dims
        self.linear1 = nn.Linear(input_features, first, bias=bias)
        self.linear2 = nn.Linear(first, second, bias=bias)
        self.readout = nn.Linear(second, classes, bias=bias)
        self.normalization1 = nn.BatchNorm1d(first) if batch_normalization else nn.Identity()
        self.normalization2 = nn.BatchNorm1d(second) if batch_normalization else nn.Identity()
        self.attention1 = make_attention(attention["variant"], attention["lambda"], neuron["threshold"])
        self.attention2 = make_attention(attention["variant"], attention["lambda"], neuron["threshold"])
        lif_kwargs = {
            "tau_ms": neuron["tau_ms"],
            "dt_ms": 10.0,
            "threshold": neuron["threshold"],
            "alpha": neuron["surrogate"]["alpha"],
            "reset": neuron["reset"],
            "detach_reset": neuron["detach_reset"],
        }
        self.lif1 = make_lif(neuron["name"], **lif_kwargs)
        self.lif2 = make_lif(neuron["name"], **lif_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.model_metadata = {
            "class": type(self).__name__,
            "input_features": input_features,
            "hidden_dims": list(hidden_dims),
            "classes": classes,
        }

    @staticmethod
    def _normalize(module, current, valid):
        if isinstance(module, nn.Identity):
            return current
        normalized = torch.zeros_like(current)
        normalized[valid] = module(current[valid])
        return normalized

    def forward(self, inputs, lengths):
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, time, features]")
        valid = torch.arange(inputs.shape[1], device=inputs.device).unsqueeze(0) < lengths.unsqueeze(1)
        current1 = self.attention1(self._normalize(self.normalization1, self.linear1(inputs), valid))
        spikes1 = self.lif1(current1, valid)
        current2 = self.attention2(self._normalize(self.normalization2, self.linear2(self.dropout(spikes1)), valid))
        spikes2 = self.lif2(current2, valid)
        logits = (self.readout(self.dropout(spikes2)) * valid.unsqueeze(-1)).sum(dim=1)
        rates = {"layer1": spike_rate(spikes1, valid), "layer2": spike_rate(spikes2, valid)}
        return logits, rates
