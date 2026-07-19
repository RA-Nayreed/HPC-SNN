import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from fedapfa.measurement.power import (
    DeviceSample,
    NvmlAdapter,
    NvmlProcessSampler,
    PowerSampler,
    ProcessPowerSampler,
)


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


class ProcessAdapter:
    uuid = "GPU-process"

    def __init__(self, marker: str, mode: str):
        self.marker = Path(marker)
        self.mode = mode
        torch_module = sys.modules.get("torch")
        cuda_initialized = bool(
            torch_module is not None and torch_module.cuda.is_initialized()
        )
        self._record(
            {
                "event": "created",
                "pid": os.getpid(),
                "start_method": multiprocessing.get_start_method(),
                "torch_cuda_initialized": cuda_initialized,
            }
        )

    def _record(self, value):
        with self.marker.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value) + "\n")

    def sample(self, timestamp, utc, interval):
        if self.mode == "sampling_error":
            raise RuntimeError("sensor")
        if self.mode == "runtime_failure":
            raise SystemExit("sampler runtime")
        power = object() if self.mode == "serialization_failure" else 100.0
        return DeviceSample(
            timestamp,
            utc,
            self.uuid,
            power,
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
        if self.mode == "shutdown_failure":
            raise RuntimeError("adapter shutdown")
        self._record({"event": "closed", "pid": os.getpid()})


class ProcessAdapterFactory:
    def __init__(self, marker: str, mode: str = "ok"):
        self.marker = marker
        self.mode = mode

    def __call__(self):
        if self.mode == "startup_failure":
            raise RuntimeError("adapter startup")
        return ProcessAdapter(self.marker, self.mode)


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


def _process_sampler(tmp_path, mode="ok", interval_ms=1000):
    return ProcessPowerSampler(
        ProcessAdapterFactory(str(tmp_path / "process-events.jsonl"), mode),
        tmp_path / "process-samples.jsonl",
        interval_ms=interval_ms,
        execution_attempt=7,
        records_per_flush=10,
        startup_timeout_seconds=5.0,
        shutdown_timeout_seconds=5.0,
    )


def test_process_sampler_uses_spawn_without_cuda_and_flushes_buffer_on_stop(tmp_path):
    sampler = _process_sampler(tmp_path)
    sampler.start()
    assert sampler.process_pid != os.getpid()
    assert sampler.process_is_alive
    time.sleep(0.1)
    sample_path = tmp_path / "process-samples.jsonl"
    assert sample_path.read_text(encoding="utf-8") == ""
    sampler.stop()
    assert not sampler.process_is_alive
    records = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines()]
    events = [
        json.loads(line)
        for line in (tmp_path / "process-events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == sampler.sample_count == 1
    assert records[0]["execution_attempt"] == 7
    assert records[0]["configured_interval_ms"] == 1000
    assert events[0]["pid"] == sampler.process_pid
    assert events[0]["start_method"] == "spawn"
    assert not events[0]["torch_cuda_initialized"]
    assert events[-1] == {"event": "closed", "pid": sampler.process_pid}


def test_process_sampler_preserves_sampling_errors_and_leaves_no_process(tmp_path):
    sampler = _process_sampler(tmp_path, "sampling_error")
    sampler.start()
    time.sleep(0.1)
    sampler.stop()
    records = [
        json.loads(line)
        for line in (tmp_path / "process-samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records
    assert sampler.error_count == len(records)
    assert all(value["sampling_error_status"] == "RuntimeError: sensor" for value in records)
    assert not sampler.process_is_alive


def test_process_sampler_propagates_startup_runtime_and_serialization_failures(tmp_path):
    startup = _process_sampler(tmp_path / "startup", "startup_failure")
    with pytest.raises(RuntimeError, match="process startup failed.*adapter startup"):
        startup.start()
    assert not startup.process_is_alive

    runtime = _process_sampler(tmp_path / "runtime", "runtime_failure")
    runtime.start()
    limit = time.monotonic() + 5
    while runtime.process_is_alive and time.monotonic() < limit:
        time.sleep(0.01)
    with pytest.raises(RuntimeError, match="process runtime failed.*sampler runtime"):
        runtime.stop()
    assert not runtime.process_is_alive

    serialization = _process_sampler(tmp_path / "serialization", "serialization_failure")
    serialization.start()
    limit = time.monotonic() + 5
    while serialization.process_is_alive and time.monotonic() < limit:
        time.sleep(0.01)
    with pytest.raises(RuntimeError, match="process serialization failed"):
        serialization.stop()
    assert not serialization.process_is_alive


def test_process_sampler_propagates_file_writing_and_shutdown_failures(tmp_path):
    blocked_path = tmp_path / "blocked"
    blocked_path.mkdir()
    writing = ProcessPowerSampler(
        ProcessAdapterFactory(str(tmp_path / "writing-events.jsonl")),
        blocked_path,
        startup_timeout_seconds=5.0,
        shutdown_timeout_seconds=5.0,
    )
    with pytest.raises(RuntimeError, match="process file-writing failed"):
        writing.start()
    assert not writing.process_is_alive

    shutdown = _process_sampler(tmp_path / "shutdown", "shutdown_failure")
    shutdown.start()
    time.sleep(0.1)
    with pytest.raises(RuntimeError, match="process shutdown failed.*adapter shutdown"):
        shutdown.stop()
    assert not shutdown.process_is_alive


def test_scientific_nvml_sampler_is_spawned_and_keeps_100_ms_default(tmp_path):
    sampler = NvmlProcessSampler("GPU-a", 1, tmp_path / "samples.jsonl")
    assert isinstance(sampler, ProcessPowerSampler)
    assert sampler._context.get_start_method() == "spawn"
    assert sampler.interval_ms == 100
    assert sampler.adapter_factory.expected_uuid == "GPU-a"
    assert sampler.adapter_factory.visible_device_count == 1
