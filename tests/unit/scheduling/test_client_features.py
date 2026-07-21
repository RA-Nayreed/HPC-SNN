from dataclasses import dataclass

import h5py
import numpy as np

from fedapfa.scheduling.base import EVENT_STRUCTURE_FEATURES
from fedapfa.scheduling.client_features import EventStructureFeatureCache, privacy_metadata_record


@dataclass
class TrainingEventDataset:
    path: str
    indices: np.ndarray
    temporal_bin_ms: float = 1.0


def test_training_event_features_are_cached_and_metadata_excludes_raw_events(tmp_path):
    path = tmp_path / "training_events.h5"
    with h5py.File(path, "w") as handle:
        spikes = handle.create_group("spikes")
        times = spikes.create_dataset("times", (3,), dtype=h5py.vlen_dtype(np.dtype("float64")))
        times[0] = np.asarray([0.0001, 0.0011])
        times[1] = np.asarray([0.0002])
        times[2] = np.asarray([0.0003, 0.0013, 0.0023])
    dataset = TrainingEventDataset(str(path), np.asarray([0, 1, 2]))
    cache = EventStructureFeatureCache()
    first = cache.features(
        "client_a",
        dataset,
        training_seed=37,
        batch_size=2,
        input_features=700,
        local_epochs=1,
        drop_last=False,
    )
    second = cache.features(
        "client_a",
        dataset,
        training_seed=47,
        batch_size=2,
        input_features=700,
        local_epochs=1,
        drop_last=False,
    )
    assert set(first.values) == set(EVENT_STRUCTURE_FEATURES)
    assert first.values["total_raw_input_events"] == 6
    assert not first.cache_hit and second.cache_hit
    records, serialized_bytes = privacy_metadata_record(first.values)
    assert serialized_bytes > 0
    assert [value["field"] for value in records] == list(EVENT_STRUCTURE_FEATURES)
    assert all(value["contains_label_information"] is False for value in records)
    assert all(value["raw_events_leave_client"] is False for value in records)
    assert {value["field"] for value in records if value["stability"] == "round_dependent"} == {
        "estimated_padded_time_bins",
        "padding_fraction",
    }
