"""Cached training-only event-structure predictors and privacy accounting."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass

import h5py
import numpy as np
import torch

from .base import EVENT_STRUCTURE_FEATURES


@dataclass(frozen=True)
class _InvariantEventFeatures:
    dataset_path: str
    indices_sha256: str
    example_count: int
    sequence_lengths: tuple[int, ...]
    total_raw_input_events: int
    input_features: int


@dataclass(frozen=True)
class FeatureExtraction:
    values: dict[str, float]
    static_lookup_seconds: float
    invariant_extraction_seconds: float
    seed_dependent_seconds: float
    cache_hit: bool


def _indices_identity(indices: np.ndarray) -> str:
    contiguous = np.asarray(indices, dtype=np.int64).reshape(-1)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _sequence_length(times: np.ndarray, temporal_bin_ms: float) -> int:
    if len(times) == 0:
        return 1
    return int(np.floor(float(np.max(times)) / (temporal_bin_ms / 1000.0))) + 1


def _ordered_positions(count: int, training_seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(training_seed)
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    return [int(value) for value in torch.randperm(count, generator=generator)]


class EventStructureFeatureCache:
    """Cache invariant event counts and recompute seed-dependent padding exactly."""

    def __init__(self) -> None:
        self._invariant: dict[str, _InvariantEventFeatures] = {}

    def _extract_invariant(self, client_id: str, dataset, input_features: int) -> _InvariantEventFeatures:
        if input_features <= 0:
            raise ValueError("input feature count must be positive")
        indices = np.asarray(getattr(dataset, "indices", None), dtype=np.int64)
        path = getattr(dataset, "path", None)
        temporal_bin_ms = getattr(dataset, "temporal_bin_ms", None)
        if indices.ndim != 1 or len(indices) == 0 or path is None or temporal_bin_ms is None:
            raise ValueError("event-structure features require resolved client training indices")
        lengths: list[int] = []
        event_count = 0
        with h5py.File(path, "r") as handle:
            if "spikes/times" not in handle:
                raise ValueError("event-structure feature source lacks spike times")
            for index in indices:
                times = np.asarray(handle["spikes/times"][int(index)], dtype=np.float64)
                event_count += len(times)
                lengths.append(_sequence_length(times, float(temporal_bin_ms)))
        value = _InvariantEventFeatures(
            dataset_path=str(path),
            indices_sha256=_indices_identity(indices),
            example_count=len(indices),
            sequence_lengths=tuple(lengths),
            total_raw_input_events=event_count,
            input_features=input_features,
        )
        self._invariant[client_id] = value
        return value

    def features(
        self,
        client_id: str,
        dataset,
        *,
        training_seed: int,
        batch_size: int,
        input_features: int,
        local_epochs: int,
        drop_last: bool,
    ) -> FeatureExtraction:
        if local_epochs != 1:
            raise ValueError("the frozen event-structure model requires one local epoch")
        if batch_size <= 0:
            raise ValueError("local batch size must be positive")
        lookup_started = time.perf_counter()
        cached = self._invariant.get(client_id)
        indices = np.asarray(getattr(dataset, "indices", None), dtype=np.int64)
        cache_hit = (
            cached is not None
            and cached.dataset_path == str(getattr(dataset, "path", None))
            and cached.indices_sha256 == _indices_identity(indices)
            and cached.input_features == input_features
        )
        static_lookup = time.perf_counter() - lookup_started
        invariant_duration = 0.0
        if not cache_hit:
            extraction_started = time.perf_counter()
            cached = self._extract_invariant(client_id, dataset, input_features)
            invariant_duration = time.perf_counter() - extraction_started

        dependent_started = time.perf_counter()
        positions = _ordered_positions(cached.example_count, training_seed)
        if drop_last:
            positions = positions[: (len(positions) // batch_size) * batch_size]
        if not positions:
            raise ValueError("event-structure client ordering produces no local training batches")
        batches = [positions[offset : offset + batch_size] for offset in range(0, len(positions), batch_size)]
        lengths = cached.sequence_lengths
        padded = sum(len(batch) * max(lengths[position] for position in batch) for batch in batches)
        valid = sum(lengths[position] for position in positions)
        events = cached.total_raw_input_events
        values = {
            "example_count": float(cached.example_count),
            "local_batch_count": float(len(batches)),
            "total_raw_input_events": float(events),
            "mean_sequence_length": float(np.mean(lengths)),
            "median_sequence_length": float(np.median(lengths)),
            "maximum_sequence_length": float(max(lengths)),
            "total_valid_time_bins": float(valid),
            "estimated_padded_time_bins": float(padded),
            "padding_fraction": float((padded - valid) / padded if padded else 0.0),
            "event_density": float(events / (valid * input_features) if valid else 0.0),
        }
        dependent_duration = time.perf_counter() - dependent_started
        if set(values) != set(EVENT_STRUCTURE_FEATURES) or any(not math.isfinite(value) for value in values.values()):
            raise RuntimeError("event-structure feature extraction produced an incompatible record")
        return FeatureExtraction(
            values=values,
            static_lookup_seconds=static_lookup,
            invariant_extraction_seconds=invariant_duration,
            seed_dependent_seconds=dependent_duration,
            cache_hit=cache_hit,
        )


def privacy_metadata_record(features: dict[str, float]) -> tuple[list[dict], int]:
    """Describe central simulation metadata without claiming privacy preservation."""

    if set(features) != set(EVENT_STRUCTURE_FEATURES):
        raise ValueError("privacy metadata requires the strict event-structure feature schema")
    round_dependent = {"estimated_padded_time_bins", "padding_fraction"}
    records = [
        {
            "field": name,
            "value": float(features[name]),
            "contains_label_information": False,
            "may_reveal_workload_or_behavior": True,
            "stability": "round_dependent" if name in round_dependent else "static",
            "cacheable": name not in round_dependent,
            "raw_events_leave_client": False,
        }
        for name in EVENT_STRUCTURE_FEATURES
    ]
    serialized = json.dumps(
        {name: float(features[name]) for name in EVENT_STRUCTURE_FEATURES},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return records, len(serialized)
