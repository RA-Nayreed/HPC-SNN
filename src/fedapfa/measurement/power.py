"""Persistent NVML sampling resolved by allocated GPU UUID."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class DeviceSample:
    monotonic_timestamp_ns: int
    utc_timestamp: str
    gpu_uuid: str
    power_watts: float | None
    gpu_utilization_percent: float | None
    memory_utilization_percent: float | None
    allocated_device_memory_bytes: int | None
    temperature_celsius: float | None
    graphics_clock_mhz: int | None
    memory_clock_mhz: int | None
    cumulative_device_energy_millijoules: int | None
    sampling_backend: str
    configured_interval_ms: int
    sampling_error_status: str | None
    execution_attempt: int = 1

    def record(self) -> dict:
        return asdict(self)


class PowerAdapter(Protocol):
    @property
    def uuid(self) -> str: ...

    def sample(self, monotonic_timestamp_ns: int, utc_timestamp: str, interval_ms: int) -> DeviceSample: ...

    def close(self) -> None: ...


def _decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


class NvmlAdapter:
    """NVML adapter whose import remains safe on machines without NVML."""

    def __init__(self, expected_uuid: str, visible_device_count: int, binding=None) -> None:
        if visible_device_count != 1:
            raise RuntimeError("resource measurement requires exactly one visible CUDA device")
        if binding is None:
            try:
                import pynvml as binding
            except (ImportError, OSError) as error:
                raise RuntimeError(
                    "resource measurement requires the nvidia-ml-py binding and an available NVML library"
                ) from error
        self.binding = binding
        try:
            binding.nvmlInit()
        except Exception as error:
            raise RuntimeError("NVML initialization failed while resource measurement was requested") from error
        self._closed = False
        handles = []
        for index in range(int(binding.nvmlDeviceGetCount())):
            handle = binding.nvmlDeviceGetHandleByIndex(index)
            if _decode(binding.nvmlDeviceGetUUID(handle)) == expected_uuid:
                handles.append(handle)
        if len(handles) != 1:
            self.close()
            raise RuntimeError("allocated CUDA UUID did not resolve to exactly one NVML device")
        self.handle = handles[0]
        self._uuid = expected_uuid

    @property
    def uuid(self) -> str:
        return self._uuid

    def _optional(self, function_name: str, *args):
        function = getattr(self.binding, function_name, None)
        if function is None:
            return None
        try:
            return function(self.handle, *args)
        except Exception:
            return None

    def sample(self, monotonic_timestamp_ns: int, utc_timestamp: str, interval_ms: int) -> DeviceSample:
        utilization = self.binding.nvmlDeviceGetUtilizationRates(self.handle)
        memory = self.binding.nvmlDeviceGetMemoryInfo(self.handle)
        power_milliwatts = self.binding.nvmlDeviceGetPowerUsage(self.handle)
        temperature = self.binding.nvmlDeviceGetTemperature(
            self.handle, self.binding.NVML_TEMPERATURE_GPU
        )
        graphics = self._optional("nvmlDeviceGetClockInfo", self.binding.NVML_CLOCK_GRAPHICS)
        memory_clock = self._optional("nvmlDeviceGetClockInfo", self.binding.NVML_CLOCK_MEM)
        energy = self._optional("nvmlDeviceGetTotalEnergyConsumption")
        return DeviceSample(
            monotonic_timestamp_ns,
            utc_timestamp,
            self.uuid,
            float(power_milliwatts) / 1000.0,
            float(utilization.gpu),
            float(utilization.memory),
            int(memory.used),
            float(temperature),
            None if graphics is None else int(graphics),
            None if memory_clock is None else int(memory_clock),
            None if energy is None else int(energy),
            "nvml",
            interval_ms,
            None,
        )

    def close(self) -> None:
        if not self._closed:
            try:
                self.binding.nvmlShutdown()
            finally:
                self._closed = True


class PowerSampler:
    """Sample on one background thread and flush each JSON line."""

    def __init__(
        self, adapter: PowerAdapter, path: str | Path, interval_ms: int = 100, execution_attempt: int = 1
    ) -> None:
        if interval_ms <= 0:
            raise ValueError("sampling interval must be positive")
        self.adapter = adapter
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.interval_ms = interval_ms
        self.execution_attempt = execution_attempt
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        self._failure: BaseException | None = None
        self.sample_count = 0
        self.error_count = 0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("power sampler has already started")
        self._handle = self.path.open("a", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, name="fedapfa-power-sampler", daemon=True)
        self._thread.start()

    def _write(self, sample: DeviceSample) -> None:
        sample = replace(sample, execution_attempt=self.execution_attempt)
        self._handle.write(json.dumps(sample.record(), sort_keys=True, allow_nan=False) + "\n")
        self._handle.flush()
        self.sample_count += 1
        if sample.sampling_error_status is not None:
            self.error_count += 1

    def _run(self) -> None:
        next_ns = time.monotonic_ns()
        interval_ns = self.interval_ms * 1_000_000
        while not self._stop.is_set():
            now_ns = time.monotonic_ns()
            utc = datetime.now(timezone.utc).isoformat()
            try:
                sample = self.adapter.sample(now_ns, utc, self.interval_ms)
            except Exception as error:
                sample = DeviceSample(
                    now_ns,
                    utc,
                    self.adapter.uuid,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "nvml",
                    self.interval_ms,
                    f"{type(error).__name__}: {error}",
                )
            try:
                self._write(sample)
            except BaseException as error:
                self._failure = error
                self._stop.set()
                break
            next_ns += interval_ns
            remaining = max(0.0, (next_ns - time.monotonic_ns()) / 1_000_000_000)
            self._stop.wait(remaining)

    def stop(self) -> None:
        self._stop.set()
        adapter_closed = False
        thread_still_running = False
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_ms / 100.0))
            if self._thread.is_alive():
                self.adapter.close()
                adapter_closed = True
                self._thread.join(timeout=max(5.0, self.interval_ms / 100.0))
                thread_still_running = self._thread.is_alive()
        cleanup_failure = None
        if self._handle is not None:
            try:
                self._handle.flush()
                os.fsync(self._handle.fileno())
            except BaseException as error:
                cleanup_failure = error
            finally:
                self._handle.close()
                self._handle = None
        if not adapter_closed:
            try:
                self.adapter.close()
            except BaseException as error:
                cleanup_failure = cleanup_failure or error
        if thread_still_running:
            raise RuntimeError("power sampler did not stop")
        if cleanup_failure is not None:
            raise RuntimeError("power sample record cleanup failed") from cleanup_failure
        if self._failure is not None:
            raise RuntimeError("power sample record write failed") from self._failure

    def __enter__(self) -> PowerSampler:
        self.start()
        return self

    def __exit__(self, _kind, _value, _traceback) -> None:
        self.stop()
