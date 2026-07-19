"""Training-index-only client features available before assignment."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass

import h5py
import numpy as np
import torch


@dataclass(frozen=True)
class StaticClientFeatures:
    example_count: int
    local_batch_count: int
    total_raw_input_events: int
    mean_sequence_length: float
    median_sequence_length: float
    maximum_sequence_length: int
    total_valid_time_bins: int
    estimated_padded_time_bins: int
    padding_fraction: float
    event_density: float
    represented_class_count: int
    label_entropy: float

    def record(self) -> dict:
        return asdict(self)


@dataclass
class ObservedClientWork:
    actual_batch_count: int = 0
    actual_presented_examples: int = 0
    actual_valid_time_bins: int = 0
    actual_padded_time_bins: int = 0
    actual_input_events: int = 0
    layer_spike_counts: dict[str, float] | None = None
    layer_neuron_time_counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.layer_spike_counts is None:
            self.layer_spike_counts = {}
        if self.layer_neuron_time_counts is None:
            self.layer_neuron_time_counts = {}

    def observe(self, batch, rates: dict[str, torch.Tensor], layer_widths: dict[str, int]) -> None:
        examples = len(batch.labels)
        valid = int(batch.valid_mask.sum()) if batch.valid_mask is not None else examples
        padded = int(batch.inputs.shape[0] * batch.inputs.shape[1]) if batch.inputs.ndim >= 3 else examples
        self.actual_batch_count += 1
        self.actual_presented_examples += examples
        self.actual_valid_time_bins += valid
        self.actual_padded_time_bins += padded
        self.actual_input_events += int(batch.inputs.detach().sum().item())
        for name, value in rates.items():
            if name not in layer_widths:
                raise ValueError(f"spike layer width is unavailable for {name}")
            count = float(value.detach()) * valid * layer_widths[name]
            self.layer_spike_counts[name] = self.layer_spike_counts.get(name, 0.0) + count
            self.layer_neuron_time_counts[name] = (
                self.layer_neuron_time_counts.get(name, 0) + valid * layer_widths[name]
            )

    def record(self) -> dict:
        padded = self.actual_padded_time_bins
        values = {
            "actual_batch_count": self.actual_batch_count,
            "actual_presented_examples": self.actual_presented_examples,
            "actual_valid_time_bins": self.actual_valid_time_bins,
            "actual_padded_time_bins": padded,
            "actual_padding_fraction": (
                (padded - self.actual_valid_time_bins) / padded if padded else 0.0
            ),
            "actual_input_events": self.actual_input_events,
        }
        for name, count in sorted(self.layer_spike_counts.items()):
            values[f"{name}_spike_count"] = count
            width_denominator = self.layer_neuron_time_counts[name]
            values[f"{name}_spike_rate"] = count / width_denominator if width_denominator else 0.0
        return values


def _sequence_length(times: np.ndarray, temporal_bin_ms: float) -> int:
    if len(times) == 0:
        return 1
    return int(np.floor(float(np.max(times)) / (temporal_bin_ms / 1000.0))) + 1


def _ordered_positions(count: int, training_seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(training_seed)
    # DataLoader consumes one value for its worker base seed before the lazy sampler runs.
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    return [int(value) for value in torch.randperm(count, generator=generator)]


def extract_static_client_features(
    dataset,
    training_seed: int,
    batch_size: int,
    input_features: int,
    local_epochs: int = 1,
    drop_last: bool = False,
) -> StaticClientFeatures:
    """Read only the dataset's resolved training indices and reproduce shuffled padding."""

    if local_epochs != 1:
        raise ValueError("resource feature extraction requires one local epoch")
    if batch_size <= 0 or input_features <= 0:
        raise ValueError("batch size and input feature count must be positive")
    indices = np.asarray(dataset.indices, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("client training indices cannot be empty")
    lengths: list[int] = []
    labels: list[int] = []
    event_counts: list[int] = []
    with h5py.File(dataset.path, "r") as handle:
        for index in indices:
            times = np.asarray(handle["spikes/times"][int(index)], dtype=np.float64)
            event_counts.append(len(times))
            lengths.append(_sequence_length(times, float(dataset.temporal_bin_ms)))
            labels.append(int(handle["labels"][int(index)]))
    positions = _ordered_positions(len(indices), training_seed)
    if drop_last:
        positions = positions[: (len(positions) // batch_size) * batch_size]
    batches = [positions[offset : offset + batch_size] for offset in range(0, len(positions), batch_size)]
    padded = sum(len(batch) * max(lengths[position] for position in batch) for batch in batches)
    valid = sum(lengths[position] for position in positions)
    counts = Counter(labels)
    entropy = -sum((count / len(labels)) * math.log(count / len(labels)) for count in counts.values())
    events = sum(event_counts)
    return StaticClientFeatures(
        len(indices),
        len(batches),
        events,
        float(np.mean(lengths)),
        float(np.median(lengths)),
        max(lengths),
        valid,
        padded,
        (padded - valid) / padded if padded else 0.0,
        events / (valid * input_features) if valid else 0.0,
        len(counts),
        entropy,
    )


FEATURE_AVAILABILITY = {
    "before_any_client_execution": [
        "example_count",
        "local_batch_count",
        "total_raw_input_events",
        "mean_sequence_length",
        "median_sequence_length",
        "maximum_sequence_length",
        "total_valid_time_bins",
        "estimated_padded_time_bins",
        "padding_fraction",
        "event_density",
        "represented_class_count",
        "label_entropy",
        "communication_round",
        "dataset_identity",
        "model_identity",
        "parameter_count",
    ],
    "after_previous_observations": [
        "has_historical_observations",
        "historical_observation_count",
        "previous_wall_duration",
        "previous_gross_energy",
        "previous_idle_adjusted_energy",
        "previous_layer_spike_rates",
        "previous_spikes_per_example",
        "exponentially_weighted_duration",
        "exponentially_weighted_energy",
        "exponentially_weighted_spike_rates",
    ],
    "after_current_execution": [
        "actual_batch_count",
        "actual_presented_examples",
        "actual_valid_time_bins",
        "actual_padded_time_bins",
        "actual_padding_fraction",
        "actual_input_events",
        "layer_spike_counts",
        "layer_spike_rates",
        "client_wall_time",
        "data_wait_time",
        "cuda_event_time",
        "residual_host_time",
        "gross_energy",
        "idle_adjusted_energy",
        "peak_allocated_cuda_memory",
        "peak_reserved_cuda_memory",
    ],
}
