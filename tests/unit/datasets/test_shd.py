import h5py
import numpy as np
import pytest

from fedapfa.datasets.shd import EventAudioDataset
from fedapfa.datasets.validation import DatasetExpectation, DatasetValidationError, validate_hdf5


def test_hdf5_reading_and_labels(event_file):
    assert validate_hdf5(event_file, DatasetExpectation(4, 2)) == {"examples": 4, "classes": 2, "raw_channels": 700}
    dataset = EventAudioDataset(event_file, expectation=DatasetExpectation(4, 2))
    sequence, label = dataset[1]
    assert sequence.shape[1] == 140 and label == 1
    assert dataset.labels().tolist() == [0, 1, 0, 1]


def test_official_count_is_executable(event_file):
    with pytest.raises(DatasetValidationError, match="expected 8156"):
        validate_hdf5(event_file, DatasetExpectation(8156, 20))


def test_missing_and_invalid_structure(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_hdf5(tmp_path / "missing.h5")
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("labels", data=[0])
    with pytest.raises(DatasetValidationError, match="required"):
        validate_hdf5(path)


def test_hdf5_rejects_mismatched_event_arrays(tmp_path):
    path = tmp_path / "mismatched.h5"
    with h5py.File(path, "w") as handle:
        spikes = handle.create_group("spikes")
        times = spikes.create_dataset("times", (1,), dtype=h5py.vlen_dtype(np.float64))
        units = spikes.create_dataset("units", (1,), dtype=h5py.vlen_dtype(np.int64))
        times[0] = np.array([0.1])
        units[0] = np.array([], dtype=np.int64)
        handle.create_dataset("labels", data=[0])
    with pytest.raises(DatasetValidationError, match="mismatched event arrays"):
        validate_hdf5(path, DatasetExpectation(1, 1))
