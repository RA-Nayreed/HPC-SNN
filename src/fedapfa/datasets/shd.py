"""Lazy, worker-safe SHD/SSC event-audio dataset implementation."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset

from .temporal_binning import bin_events
from .validation import OFFICIAL_EXPECTATIONS, DatasetExpectation, validate_hdf5


class EventAudioDataset(Dataset):
    fedapfa_batch_kind = "event_sequence"

    def __init__(
        self,
        path: str | Path,
        indices=None,
        temporal_bin_ms: float = 10.0,
        frequency_bin_factor: int = 5,
        validate: bool = True,
        expectation: DatasetExpectation | None = None,
    ):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if validate:
            validate_hdf5(self.path, expectation or OFFICIAL_EXPECTATIONS.get(self.path.name))
        with h5py.File(self.path, "r") as handle:
            self.size = len(handle["labels"])
        self.indices = np.arange(self.size, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        if np.any(self.indices < 0) or np.any(self.indices >= self.size):
            raise IndexError("dataset indices are out of range")
        self.temporal_bin_ms = temporal_bin_ms
        self.frequency_bin_factor = frequency_bin_factor
        self._file = None
        self._pid = None

    def _handle(self):
        pid = os.getpid()
        if self._file is None or self._pid != pid:
            if self._file is not None:
                self._file.close()
            self._file = h5py.File(self.path, "r")
            self._pid = pid
        return self._file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        handle = self._handle()
        index = int(self.indices[item])
        return bin_events(
            handle["spikes/times"][index],
            handle["spikes/units"][index],
            self.temporal_bin_ms,
            self.frequency_bin_factor,
        ), int(handle["labels"][index])

    def labels(self) -> np.ndarray:
        with h5py.File(self.path, "r") as handle:
            return np.asarray(handle["labels"][:], dtype=np.int64)[self.indices]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_pid"] = None
        return state

    def metadata(self):
        return validate_hdf5(self.path, OFFICIAL_EXPECTATIONS.get(self.path.name))


class SHDDataset(EventAudioDataset):
    pass
