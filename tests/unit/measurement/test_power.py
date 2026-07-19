import json
import time
from types import SimpleNamespace

import pytest

from fedapfa.measurement.power import DeviceSample, NvmlAdapter, PowerSampler


class Binding:
    NVML_TEMPERATURE_GPU = 1
    NVML_CLOCK_GRAPHICS = 2
    NVML_CLOCK_MEM = 3

    def __init__(self, uuids=("GPU-a", "GPU-b")):
        self.uuids = uuids
        self.shutdown = False

    def nvmlInit(self):
        return None

    def nvmlShutdown(self):
        self.shutdown = True

    def nvmlDeviceGetCount(self):
        return len(self.uuids)

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetUUID(self, handle):
        return self.uuids[handle].encode()

    def nvmlDeviceGetUtilizationRates(self, _handle):
        return SimpleNamespace(gpu=42, memory=21)

    def nvmlDeviceGetMemoryInfo(self, _handle):
        return SimpleNamespace(used=4096)

    def nvmlDeviceGetPowerUsage(self, _handle):
        return 125000

    def nvmlDeviceGetTemperature(self, _handle, _sensor):
        return 47

    def nvmlDeviceGetClockInfo(self, _handle, clock):
        return 1200 if clock == self.NVML_CLOCK_GRAPHICS else 1500

    def nvmlDeviceGetTotalEnergyConsumption(self, _handle):
        return 9000


class Adapter:
    uuid = "GPU-a"

    def __init__(self, fail=False):
        self.fail = fail
        self.closed = False

    def sample(self, timestamp, utc, interval):
        if self.fail:
            raise RuntimeError("sensor")
        return DeviceSample(
            timestamp,
            utc,
            self.uuid,
            100.0,
            10.0,
            5.0,
            1024,
            40.0,
            None,
            None,
            None,
            "nvml",
            interval,
            None,
        )

    def close(self):
        self.closed = True


def test_nvml_resolution_uses_uuid_and_requires_one_visible_device():
    binding = Binding()
    adapter = NvmlAdapter("GPU-b", 1, binding)
    assert adapter.uuid == "GPU-b"
    sample = adapter.sample(10, "utc", 100)
    assert sample.power_watts == 125.0
    assert sample.allocated_device_memory_bytes == 4096
    assert sample.cumulative_device_energy_millijoules == 9000
    adapter.close()
    assert binding.shutdown
    with pytest.raises(RuntimeError, match="exactly one visible"):
        NvmlAdapter("GPU-a", 2, Binding())


def test_nvml_uuid_must_resolve_exactly_once():
    with pytest.raises(RuntimeError, match="exactly one NVML"):
        NvmlAdapter("GPU-z", 1, Binding())
    with pytest.raises(RuntimeError, match="exactly one NVML"):
        NvmlAdapter("GPU-a", 1, Binding(("GPU-a", "GPU-a")))


@pytest.mark.parametrize("fail", [False, True])
def test_sampler_flushes_and_closes_after_shutdown_or_sampling_exception(tmp_path, fail):
    path = tmp_path / "samples.jsonl"
    adapter = Adapter(fail=fail)
    sampler = PowerSampler(adapter, path, interval_ms=1, execution_attempt=3)
    sampler.start()
    limit = time.monotonic() + 2
    while sampler.sample_count < 1 and time.monotonic() < limit:
        time.sleep(0.001)
    sampler.stop()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records and records[0]["execution_attempt"] == 3
    assert (records[0]["sampling_error_status"] is not None) is fail
    assert (sampler.error_count > 0) is fail
    if fail:
        assert all(value["sampling_error_status"] is not None for value in records)
    assert adapter.closed
