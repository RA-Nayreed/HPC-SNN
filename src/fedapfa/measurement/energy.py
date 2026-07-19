"""Boundary-aware trapezoidal integration of device power."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .power import DeviceSample


@dataclass(frozen=True)
class EnergyEstimate:
    start_ns: int
    end_ns: int
    sample_count: int
    coverage_seconds: float
    gross_energy_joules: float
    idle_adjusted_energy_joules: float
    idle_baseline_watts: float
    cumulative_energy_crosscheck_joules: float | None

    def record(self) -> dict:
        return asdict(self)


def _validate_samples(samples: list[DeviceSample]) -> None:
    if len(samples) < 2:
        raise ValueError("energy integration requires at least two power samples")
    timestamps = [value.monotonic_timestamp_ns for value in samples]
    if any(right == left for left, right in zip(timestamps, timestamps[1:], strict=False)):
        raise ValueError("power samples contain a duplicated timestamp")
    if any(right < left for left, right in zip(timestamps, timestamps[1:], strict=False)):
        raise ValueError("power sample timestamps are not monotonic")
    for sample in samples:
        if sample.sampling_error_status is not None:
            raise ValueError("power samples contain a sampling error")
        if sample.power_watts is None or not math.isfinite(sample.power_watts) or sample.power_watts < 0:
            raise ValueError("power samples require finite non-negative power")
    if len({value.gpu_uuid for value in samples}) != 1:
        raise ValueError("power samples contain more than one GPU UUID")


def _interpolate(left: DeviceSample, right: DeviceSample, timestamp_ns: int) -> tuple[float, float | None]:
    span = right.monotonic_timestamp_ns - left.monotonic_timestamp_ns
    ratio = (timestamp_ns - left.monotonic_timestamp_ns) / span
    power = left.power_watts + (right.power_watts - left.power_watts) * ratio
    if left.cumulative_device_energy_millijoules is None or right.cumulative_device_energy_millijoules is None:
        energy = None
    else:
        energy = left.cumulative_device_energy_millijoules + (
            right.cumulative_device_energy_millijoules - left.cumulative_device_energy_millijoules
        ) * ratio
    return power, energy


def integrate_energy(
    samples: list[DeviceSample],
    start_ns: int,
    end_ns: int,
    idle_baseline_watts: float,
    configured_interval_ms: int = 100,
    maximum_gap_multiplier: float = 2.5,
) -> EnergyEstimate:
    """Integrate one interval, interpolating power at both boundaries."""

    if end_ns <= start_ns:
        raise ValueError("energy interval must have positive duration")
    if idle_baseline_watts < 0 or not math.isfinite(idle_baseline_watts):
        raise ValueError("idle baseline must be finite and non-negative")
    ordered = list(samples)
    _validate_samples(ordered)
    leading = [value for value in ordered if value.monotonic_timestamp_ns <= start_ns]
    trailing = [value for value in ordered if value.monotonic_timestamp_ns >= end_ns]
    if not leading:
        raise ValueError("energy interval has no leading power sample")
    if not trailing:
        raise ValueError("energy interval has no trailing power sample")
    before = leading[-1]
    after_start = next(
        (value for value in ordered if value.monotonic_timestamp_ns >= start_ns), None
    )
    before_end = next(
        (value for value in reversed(ordered) if value.monotonic_timestamp_ns <= end_ns), None
    )
    after = trailing[0]
    if after_start is None or before_end is None:
        raise ValueError("energy interval boundaries are not covered")
    maximum_gap_ns = configured_interval_ms * 1_000_000 * maximum_gap_multiplier
    relevant = [
        value
        for value in ordered
        if before.monotonic_timestamp_ns <= value.monotonic_timestamp_ns <= after.monotonic_timestamp_ns
    ]
    if any(
        right.monotonic_timestamp_ns - left.monotonic_timestamp_ns > maximum_gap_ns
        for left, right in zip(relevant, relevant[1:], strict=False)
    ):
        raise ValueError("power sample gap exceeds the configured coverage limit")
    start_power, start_energy = _interpolate(before, after_start, start_ns) if before != after_start else (
        before.power_watts,
        before.cumulative_device_energy_millijoules,
    )
    end_power, end_energy = _interpolate(before_end, after, end_ns) if before_end != after else (
        after.power_watts,
        after.cumulative_device_energy_millijoules,
    )
    points = [(start_ns, float(start_power), start_energy)]
    points.extend(
        (
            value.monotonic_timestamp_ns,
            float(value.power_watts),
            value.cumulative_device_energy_millijoules,
        )
        for value in ordered
        if start_ns < value.monotonic_timestamp_ns < end_ns
    )
    points.append((end_ns, float(end_power), end_energy))
    gross = dynamic = 0.0
    for left, right in zip(points, points[1:], strict=False):
        duration_seconds = (right[0] - left[0]) / 1_000_000_000
        mean_power = (left[1] + right[1]) / 2.0
        gross += mean_power * duration_seconds
        dynamic += max(mean_power - idle_baseline_watts, 0.0) * duration_seconds
    cumulative = None
    if start_energy is not None and end_energy is not None:
        cumulative = (float(end_energy) - float(start_energy)) / 1000.0
        if cumulative < 0:
            raise ValueError("cumulative hardware energy decreased across the interval")
    return EnergyEstimate(
        start_ns,
        end_ns,
        len(relevant),
        (end_ns - start_ns) / 1_000_000_000,
        gross,
        dynamic,
        idle_baseline_watts,
        cumulative,
    )
