from dataclasses import replace

import pytest

from fedapfa.measurement.energy import integrate_energy
from fedapfa.measurement.power import DeviceSample


def _sample(milliseconds, watts, cumulative=None):
    return DeviceSample(
        monotonic_timestamp_ns=milliseconds * 1_000_000,
        utc_timestamp="2026-07-19T00:00:00+00:00",
        gpu_uuid="GPU-a",
        power_watts=watts,
        gpu_utilization_percent=50.0,
        memory_utilization_percent=25.0,
        allocated_device_memory_bytes=1024,
        temperature_celsius=45.0,
        graphics_clock_mhz=1200,
        memory_clock_mhz=1500,
        cumulative_device_energy_millijoules=cumulative,
        sampling_backend="nvml",
        configured_interval_ms=100,
        sampling_error_status=None,
    )


def test_exact_trapezoidal_and_idle_adjusted_integrals():
    samples = [_sample(0, 10.0), _sample(100, 20.0), _sample(200, 30.0)]
    value = integrate_energy(samples, 0, 200_000_000, idle_baseline_watts=12.0)
    assert value.gross_energy_joules == pytest.approx(4.0)
    assert value.idle_adjusted_energy_joules == pytest.approx(1.6)
    assert value.coverage_seconds == pytest.approx(0.2)
    assert value.sample_count == 3


def test_boundary_interpolation_and_hardware_crosscheck():
    samples = [_sample(0, 10.0, 1000), _sample(100, 20.0, 2500), _sample(200, 30.0, 5000)]
    value = integrate_energy(samples, 50_000_000, 150_000_000, idle_baseline_watts=0.0)
    assert value.gross_energy_joules == pytest.approx(2.0)
    assert value.cumulative_energy_crosscheck_joules == pytest.approx(2.0)
    assert value.sample_count == 3


@pytest.mark.parametrize(
    ("samples", "start", "end", "message"),
    [
        ([_sample(100, 10.0), _sample(200, 10.0)], 50_000_000, 150_000_000, "leading"),
        ([_sample(0, 10.0), _sample(100, 10.0)], 50_000_000, 150_000_000, "trailing"),
        ([_sample(0, 10.0), _sample(300, 10.0)], 0, 300_000_000, "gap"),
        ([_sample(0, 10.0), _sample(0, 11.0)], 0, 1, "duplicated"),
        ([_sample(100, 10.0), _sample(0, 11.0)], 0, 100_000_000, "monotonic"),
        ([_sample(0, -1.0), _sample(100, 10.0)], 0, 100_000_000, "non-negative"),
    ],
)
def test_invalid_power_coverage_is_rejected(samples, start, end, message):
    with pytest.raises(ValueError, match=message):
        integrate_energy(samples, start, end, idle_baseline_watts=0.0)


def test_sampling_error_and_decreasing_hardware_energy_are_rejected():
    samples = [_sample(0, 10.0, 2000), _sample(100, 10.0, 1000)]
    with pytest.raises(ValueError, match="decreased"):
        integrate_energy(samples, 0, 100_000_000, idle_baseline_watts=0.0)
    samples[1] = replace(samples[1], cumulative_device_energy_millijoules=3000, sampling_error_status="read")
    with pytest.raises(ValueError, match="sampling error"):
        integrate_energy(samples, 0, 100_000_000, idle_baseline_watts=0.0)
