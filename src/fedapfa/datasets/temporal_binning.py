"""Shared event preprocessing with output layout [time, features]."""

import numpy as np
import torch

from .cochlear_binning import bin_cochlear_channels


def bin_events(
    times: np.ndarray,
    units: np.ndarray,
    temporal_bin_ms: float = 10.0,
    frequency_bin_factor: int = 5,
    raw_channels: int = 700,
) -> torch.Tensor:
    times = np.asarray(times, dtype=np.float64)
    units = np.asarray(units)
    if times.ndim != 1 or units.ndim != 1:
        raise ValueError("times and units must be one-dimensional")
    if len(times) != len(units):
        raise ValueError("times and units must have matching lengths")
    if temporal_bin_ms <= 0:
        raise ValueError("temporal-bin width must be positive")
    if np.any(~np.isfinite(times)) or np.any(times < 0):
        raise ValueError("event times must be finite and non-negative")
    channels = bin_cochlear_channels(units, frequency_bin_factor, raw_channels)
    features = raw_channels // frequency_bin_factor
    if len(times) == 0:
        return torch.zeros((1, features), dtype=torch.float32)
    time_indices = np.floor(times / (temporal_bin_ms / 1000.0)).astype(np.int64)
    output = torch.zeros((int(time_indices.max()) + 1, features), dtype=torch.float32)
    output.index_put_(
        (torch.from_numpy(time_indices), torch.from_numpy(channels)), torch.ones(len(times)), accumulate=True
    )
    return output
