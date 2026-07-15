"""Optional DCLS delay network using the official dcls package without fallback."""

from __future__ import annotations

from importlib.util import find_spec

import torch
from torch import nn

from fedapfa.metrics.spike_rate import spike_rate
from fedapfa.neurons.lif import make_lif


class DCLSUnavailableError(RuntimeError):
    pass


def dcls_available() -> bool:
    return find_spec("DCLS") is not None


class DCLSSHDSNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not dcls_available():
            raise DCLSUnavailableError(
                "dcls_shd requires dcls==0.1.1, which is not installed; no ordinary-LIF fallback is permitted"
            )
        try:
            from DCLS.construct.modules import Dcls1d
        except Exception as error:
            raise DCLSUnavailableError(f"dcls==0.1.1 import failed on this platform: {error}") from error
        model = config["model"]
        dataset = config["dataset"]
        first, second = model["hidden_dims"]
        delay_bins = round(model["maximum_delay_ms"] / dataset["temporal_bin_ms"])
        kernel_size = delay_bins if delay_bins % 2 else delay_bins + 1
        try:
            self.delay1 = Dcls1d(
                dataset["input_features"],
                first,
                kernel_count=1,
                dilated_kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=model["bias"],
            )
            self.delay2 = Dcls1d(
                first,
                second,
                kernel_count=1,
                dilated_kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=model["bias"],
            )
        except Exception as error:
            raise DCLSUnavailableError(f"dcls==0.1.1 layer construction failed: {error}") from error
        self.normalization1 = nn.BatchNorm1d(first) if model["batch_normalization"] else nn.Identity()
        self.normalization2 = nn.BatchNorm1d(second) if model["batch_normalization"] else nn.Identity()
        self.dropout = nn.Dropout(model["dropout"])
        self.readout = nn.Linear(second, dataset["classes"], bias=model["bias"])
        neuron = model["neuron"]
        kwargs = {
            "tau_ms": neuron["tau_ms"],
            "dt_ms": dataset["temporal_bin_ms"],
            "threshold": neuron["threshold"],
            "alpha": neuron["surrogate"]["alpha"],
            "reset": neuron["reset"],
            "detach_reset": neuron["detach_reset"],
        }
        self.lif1 = make_lif(neuron["name"], **kwargs)
        self.lif2 = make_lif(neuron["name"], **kwargs)
        self.model_metadata = {
            "class": type(self).__name__,
            "dcls_version": "0.1.1",
            "maximum_delay_ms": model["maximum_delay_ms"],
            "hidden_dims": list(model["hidden_dims"]),
        }

    def delay_parameters(self):
        return [module.P for module in (self.delay1, self.delay2)]

    def forward(self, inputs, lengths):
        valid = torch.arange(inputs.shape[1], device=inputs.device).unsqueeze(0) < lengths.unsqueeze(1)
        current1 = self.delay1(inputs.transpose(1, 2)).transpose(1, 2)
        normalized1 = torch.zeros_like(current1)
        normalized1[valid] = self.normalization1(current1[valid])
        current1 = normalized1
        spikes1 = self.lif1(current1, valid)
        current2 = self.delay2(self.dropout(spikes1).transpose(1, 2)).transpose(1, 2)
        normalized2 = torch.zeros_like(current2)
        normalized2[valid] = self.normalization2(current2[valid])
        current2 = normalized2
        spikes2 = self.lif2(current2, valid)
        logits = (self.readout(self.dropout(spikes2)) * valid.unsqueeze(-1)).sum(1)
        rates = {"layer1": spike_rate(spikes1, valid), "layer2": spike_rate(spikes2, valid)}
        return logits, rates
