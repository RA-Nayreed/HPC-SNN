"""Deterministic 700-to-140 cochlear channel mapping."""

import numpy as np


def bin_cochlear_channels(units: np.ndarray, factor: int = 5, raw_channels: int = 700) -> np.ndarray:
    units = np.asarray(units)
    if units.ndim != 1:
        raise ValueError("cochlear units must be one-dimensional")
    if factor <= 0 or raw_channels % factor:
        raise ValueError("frequency-bin factor must divide raw channel count")
    if not np.issubdtype(units.dtype, np.number) or np.any(~np.isfinite(units)):
        raise ValueError("cochlear units must be finite numbers")
    if np.any(units != np.floor(units)):
        raise ValueError("cochlear units must be integer channel IDs")
    if np.any(units < 0) or np.any(units >= raw_channels):
        raise ValueError(f"cochlear units must be in [0, {raw_channels - 1}]")
    return (units.astype(np.int64, copy=False) // factor).astype(np.int64, copy=False)
