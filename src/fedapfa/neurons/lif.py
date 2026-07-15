"""Auditable Euler and SpikingJelly-compatible LIF recurrences."""

from __future__ import annotations

import torch
from torch import nn

from .surrogate import atan_spike


class BaseLIF(nn.Module):
    def __init__(self, tau_ms=10.05, dt_ms=10.0, threshold=1.0, alpha=5.0, reset="subtract", detach_reset=True):
        super().__init__()
        if tau_ms <= 0 or dt_ms <= 0 or threshold <= 0:
            raise ValueError("LIF constants must be positive")
        if reset not in {"subtract", "zero"}:
            raise ValueError("reset must be subtract or zero")
        if not detach_reset:
            raise ValueError("only detached reset is supported")
        self.tau_ms = float(tau_ms)
        self.dt_ms = float(dt_ms)
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.reset = reset
        self.membrane = None

    def reset_state(self):
        self.membrane = None

    def charge(self, membrane, current):
        raise NotImplementedError

    def forward(self, current: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        if current.ndim != 3:
            raise ValueError("LIF input must have shape [batch, time, neurons]")
        self.reset_state()
        membrane = torch.zeros_like(current[:, 0])
        outputs = []
        if valid_mask is None:
            valid_mask = torch.ones(current.shape[:2], dtype=torch.bool, device=current.device)
        for time_index in range(current.shape[1]):
            charged = self.charge(membrane, current[:, time_index])
            spike = atan_spike(charged - self.threshold, self.alpha)
            reset = (
                charged - spike.detach() * self.threshold
                if self.reset == "subtract"
                else charged * (1 - spike.detach())
            )
            valid = valid_mask[:, time_index].unsqueeze(-1)
            membrane = torch.where(valid, reset, membrane)
            outputs.append(spike * valid)
        self.membrane = membrane.detach()
        return torch.stack(outputs, dim=1)


class EulerLIF(BaseLIF):
    """u[t] = (1 - dt/tau) u[t-1] + I[t]."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.tau_ms <= self.dt_ms:
            raise ValueError("EulerLIF requires tau_ms > dt_ms")

    def charge(self, membrane, current):
        return (1 - self.dt_ms / self.tau_ms) * membrane + current


class SpikingJellyLIF(BaseLIF):
    """SpikingJelly LIFNode charge with decay_input=True and soft/hard detached reset."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tau_steps = self.tau_ms / self.dt_ms
        if self.tau_steps <= 1:
            raise ValueError("SpikingJellyLIF requires tau_ms/dt_ms > 1")

    def charge(self, membrane, current):
        return membrane + (current - membrane) / self.tau_steps


LIF = EulerLIF


def make_lif(name: str, **kwargs) -> BaseLIF:
    if name == "euler_lif":
        return EulerLIF(**kwargs)
    if name == "spikingjelly_lif":
        return SpikingJellyLIF(**kwargs)
    raise ValueError(f"unknown LIF variant: {name}")
