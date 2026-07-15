"""Spike-rate calculation excluding padded timesteps."""

import torch


def spike_rate(spikes: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if spikes.ndim != 3 or valid_mask.shape != spikes.shape[:2]:
        raise ValueError("expected spikes [batch,time,neurons] and matching mask")
    denominator = valid_mask.sum() * spikes.shape[-1]
    return (spikes * valid_mask.unsqueeze(-1)).sum() / denominator.clamp_min(1)
