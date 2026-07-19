import math

import h5py
import numpy as np
import pytest

from fedapfa.datasets.shd import EventAudioDataset
from fedapfa.measurement.features import extract_static_client_features


def _events(path):
    with h5py.File(path, "w") as handle:
        spikes = handle.create_group("spikes")
        times = spikes.create_dataset("times", (4,), dtype=h5py.vlen_dtype(np.float64))
        units = spikes.create_dataset("units", (4,), dtype=h5py.vlen_dtype(np.int64))
        values = [[], [0.0, 0.009, 0.019], [0.0, 0.021], [0.0] * 40]
        for index, item in enumerate(values):
            times[index] = np.asarray(item, dtype=np.float64)
            units[index] = np.arange(len(item), dtype=np.int64)
        handle.create_dataset("labels", data=np.asarray([0, 0, 1, 9], dtype=np.int64))


def test_exact_training_index_features_and_padding(tmp_path):
    path = tmp_path / "events.h5"
    _events(path)
    dataset = EventAudioDataset(path, indices=[0, 1, 2], validate=False)
    value = extract_static_client_features(dataset, 7, batch_size=2, input_features=140)
    assert value.example_count == 3
    assert value.local_batch_count == 2
    assert value.total_raw_input_events == 5
    assert value.mean_sequence_length == pytest.approx(2.0)
    assert value.median_sequence_length == 2.0
    assert value.maximum_sequence_length == 3
    assert value.total_valid_time_bins == 6
    assert value.estimated_padded_time_bins == 7
    assert value.padding_fraction == pytest.approx(1 / 7)
    assert value.event_density == pytest.approx(5 / (6 * 140))
    assert value.represented_class_count == 2
    assert value.label_entropy == pytest.approx(-(2 / 3) * math.log(2 / 3) - (1 / 3) * math.log(1 / 3))


def test_excluded_indices_cannot_enter_features(tmp_path):
    path = tmp_path / "events.h5"
    _events(path)
    selected = EventAudioDataset(path, indices=[0, 1, 2], validate=False)
    reference = extract_static_client_features(selected, 7, 2, 140)
    with h5py.File(path, "r+") as handle:
        handle["labels"][3] = 4
        handle["spikes/times"][3] = np.asarray([0.0] * 80)
    observed = extract_static_client_features(selected, 7, 2, 140)
    assert observed == reference


def test_feature_extraction_rejects_protocol_changes(tmp_path):
    path = tmp_path / "events.h5"
    _events(path)
    dataset = EventAudioDataset(path, indices=[0, 1, 2], validate=False)
    with pytest.raises(ValueError, match="one local epoch"):
        extract_static_client_features(dataset, 7, 2, 140, local_epochs=2)
