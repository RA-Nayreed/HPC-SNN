import numpy as np
import pytest
import torch

from fedapfa.datasets.cochlear_binning import bin_cochlear_channels
from fedapfa.datasets.sequence_collation import collate_event_sequences
from fedapfa.datasets.temporal_binning import bin_events


def test_cochlear_boundaries():
    assert bin_cochlear_channels(np.array([0, 4, 5, 699])).tolist() == [0, 0, 1, 139]


def test_temporal_boundaries_and_counts():
    output = bin_events(np.array([0.009999, 0.010000, 0.010001]), np.array([0, 0, 0]))
    assert output.shape == (2, 140)
    assert output[:, 0].tolist() == [1, 2]


def test_empty_event_is_one_documented_zero_bin():
    assert torch.equal(bin_events(np.array([]), np.array([], dtype=int)), torch.zeros(1, 140))


@pytest.mark.parametrize(
    "times,units",
    [
        (np.array([-1.0]), np.array([0])),
        (np.array([0.0]), np.array([700])),
        (np.array([0.0]), np.array([4.5])),
        (np.array([0.0]), np.array([np.nan])),
        (np.array([0.0, 1.0]), np.array([0])),
    ],
)
def test_invalid_events(times, units):
    with pytest.raises(ValueError):
        bin_events(times, units)


def test_collation_lengths_and_mask():
    batch = collate_event_sequences([(torch.ones(2, 140), 1), (torch.ones(1, 140), 2)])
    assert batch.inputs.shape == (2, 2, 140)
    assert batch.lengths.tolist() == [2, 1]
    assert batch.valid_mask.tolist() == [[True, True], [True, False]]
    assert batch.inputs[1, 1].sum() == 0


def test_empty_malformed_batch():
    with pytest.raises(ValueError):
        collate_event_sequences([])
