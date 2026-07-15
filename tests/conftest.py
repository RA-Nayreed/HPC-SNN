from pathlib import Path

import h5py
import numpy as np
import pytest


def write_event_h5(path: Path, labels=(0, 1, 0, 1)) -> Path:
    dtype = h5py.vlen_dtype(np.dtype("float64"))
    unit_dtype = h5py.vlen_dtype(np.dtype("int64"))
    with h5py.File(path, "w") as handle:
        spikes = handle.create_group("spikes")
        times = spikes.create_dataset("times", (len(labels),), dtype=dtype)
        units = spikes.create_dataset("units", (len(labels),), dtype=unit_dtype)
        for index in range(len(labels)):
            times[index] = np.asarray([0.001, 0.010, 0.011 + index * 0.001])
            units[index] = np.asarray([0, 5, 699])
        handle.create_dataset("labels", data=np.asarray(labels, dtype=np.int64))
    return path


@pytest.fixture
def event_file(tmp_path):
    return write_event_h5(tmp_path / "events.h5")
