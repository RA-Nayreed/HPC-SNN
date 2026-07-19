"""Persistent NVML sampling resolved by allocated GPU UUID."""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
import traceback
from collections.abc import Callable
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
    """Injectable in-process sampler retained for deterministic unit tests."""

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


@dataclass(frozen=True)
class _NvmlAdapterFactory:
    expected_uuid: str
    visible_device_count: int

    def __call__(self) -> PowerAdapter:
        return NvmlAdapter(self.expected_uuid, self.visible_device_count)


def _failure_record(phase: str, error: BaseException) -> dict:
    return {
        "kind": "failed",
        "phase": phase,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
    }


def _flush_sample_buffer(handle, buffer: list[str]) -> None:
    if not buffer:
        return
    handle.writelines(buffer)
    handle.flush()
    buffer.clear()


def _run_sampler_process(
    adapter_factory: Callable[[], PowerAdapter],
    path: str,
    interval_ms: int,
    execution_attempt: int,
    records_per_flush: int,
    stop_event,
    status_connection,
) -> None:
    """Run without importing Torch or calling any CUDA API."""

    adapter = None
    handle = None
    buffer: list[str] = []
    sample_count = 0
    error_count = 0
    failure = None
    started = False
    phase = "startup"
    try:
        adapter = adapter_factory()
        phase = "file-writing"
        handle = Path(path).open("a", encoding="utf-8")
        status_connection.send({"kind": "started", "pid": os.getpid()})
        started = True
        interval_ns = interval_ms * 1_000_000
        next_ns = time.monotonic_ns()
        phase = "runtime"
        while not stop_event.is_set():
            now_ns = time.monotonic_ns()
            utc = datetime.now(timezone.utc).isoformat()
            try:
                sample = adapter.sample(now_ns, utc, interval_ms)
            except Exception as error:
                sample = DeviceSample(
                    now_ns,
                    utc,
                    adapter.uuid,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "nvml",
                    interval_ms,
                    f"{type(error).__name__}: {error}",
                )
            sample = replace(sample, execution_attempt=execution_attempt)
            phase = "serialization"
            buffer.append(json.dumps(sample.record(), sort_keys=True, allow_nan=False) + "\n")
            sample_count += 1
            if sample.sampling_error_status is not None:
                error_count += 1
            if len(buffer) >= records_per_flush:
                phase = "file-writing"
                _flush_sample_buffer(handle, buffer)
            phase = "runtime"
            next_ns += interval_ns
            remaining = max(0.0, (next_ns - time.monotonic_ns()) / 1_000_000_000)
            stop_event.wait(remaining)
    except BaseException as error:
        failure = _failure_record(phase, error)
    finally:
        cleanup_failures = []
        if handle is not None:
            try:
                _flush_sample_buffer(handle, buffer)
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException as error:
                cleanup_failures.append(_failure_record("shutdown-file-writing", error))
            try:
                handle.close()
            except BaseException as error:
                cleanup_failures.append(_failure_record("shutdown-file-writing", error))
        if adapter is not None:
            try:
                adapter.close()
            except BaseException as error:
                cleanup_failures.append(_failure_record("shutdown", error))
        if failure is None and cleanup_failures:
            failure = cleanup_failures.pop(0)
        if failure is not None:
            if cleanup_failures:
                failure["cleanup_failures"] = cleanup_failures
            failure["started"] = started
            failure["sample_count"] = sample_count
            failure["error_count"] = error_count
            message = failure
        else:
            message = {
                "kind": "stopped",
                "sample_count": sample_count,
                "error_count": error_count,
            }
        try:
            status_connection.send(message)
        finally:
            status_connection.close()


class ProcessPowerSampler:
    """Sample and buffer records in a spawned operating-system process."""

    def __init__(
        self,
        adapter_factory: Callable[[], PowerAdapter],
        path: str | Path,
        interval_ms: int = 100,
        execution_attempt: int = 1,
        *,
        records_per_flush: int = 10,
        startup_timeout_seconds: float = 30.0,
        shutdown_timeout_seconds: float = 10.0,
    ) -> None:
        if interval_ms <= 0:
            raise ValueError("sampling interval must be positive")
        if records_per_flush <= 1:
            raise ValueError("process sampling must buffer more than one record per flush")
        self.adapter_factory = adapter_factory
        self.path = Path(path)
        self.interval_ms = interval_ms
        self.execution_attempt = execution_attempt
        self.records_per_flush = records_per_flush
        self.startup_timeout_seconds = startup_timeout_seconds
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self.sample_count = 0
        self.error_count = 0
        self._context = multiprocessing.get_context("spawn")
        self._stop = None
        self._process = None
        self._status_receiver = None
        self._status_sender = None
        self._started = False

    @property
    def process_pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    @property
    def process_is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def _terminate_after_failure(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._process is not None and self._process.pid is not None:
            self._process.join(timeout=self.shutdown_timeout_seconds)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=self.shutdown_timeout_seconds)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=self.shutdown_timeout_seconds)
        still_alive = self._process is not None and self._process.is_alive()
        try:
            for name in ("_status_receiver", "_status_sender"):
                connection = getattr(self, name)
                if connection is not None:
                    connection.close()
                    setattr(self, name, None)
            self._started = False
        finally:
            if still_alive:
                raise RuntimeError("power sampler process could not be terminated")

    @staticmethod
    def _raised_failure(status: dict) -> RuntimeError:
        phase = status.get("phase", "unknown")
        kind = status.get("error_type", "Exception")
        message = status.get("error_message", "unknown sampler-process failure")
        error = RuntimeError(f"power sampler process {phase} failed: {kind}: {message}")
        error.add_note(status.get("traceback", ""))
        return error

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("power sampler has already started")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stop = self._context.Event()
        self._status_receiver, self._status_sender = self._context.Pipe(duplex=False)
        self._process = self._context.Process(
            target=_run_sampler_process,
            name="fedapfa-power-sampler",
            args=(
                self.adapter_factory,
                str(self.path),
                self.interval_ms,
                self.execution_attempt,
                self.records_per_flush,
                self._stop,
                self._status_sender,
            ),
            daemon=False,
        )
        try:
            self._process.start()
            self._status_sender.close()
            self._status_sender = None
            deadline = time.monotonic() + self.startup_timeout_seconds
            while time.monotonic() < deadline:
                if self._status_receiver.poll(0.05):
                    status = self._status_receiver.recv()
                    if status.get("kind") == "started":
                        self._started = True
                        return
                    if status.get("kind") == "failed":
                        raise self._raised_failure(status)
                    raise RuntimeError("power sampler process returned an invalid startup status")
                if self._process.exitcode is not None:
                    raise RuntimeError(
                        f"power sampler process exited during startup with code {self._process.exitcode}"
                    )
            raise RuntimeError("power sampler process startup timed out")
        except BaseException:
            self._terminate_after_failure()
            raise

    def stop(self) -> None:
        if not self._started or self._process is None or self._stop is None:
            raise RuntimeError("power sampler was not started")
        self._stop.set()
        try:
            self._process.join(timeout=self.shutdown_timeout_seconds)
        except BaseException:
            self._terminate_after_failure()
            raise
        clean_shutdown = not self._process.is_alive()
        if not clean_shutdown:
            self._process.terminate()
            self._process.join(timeout=self.shutdown_timeout_seconds)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=self.shutdown_timeout_seconds)
        still_alive = self._process.is_alive()
        statuses = []
        try:
            while self._status_receiver.poll():
                statuses.append(self._status_receiver.recv())
        except EOFError:
            pass
        finally:
            self._status_receiver.close()
            self._status_receiver = None
            self._started = False
        failures = [value for value in statuses if value.get("kind") == "failed"]
        stopped = next((value for value in statuses if value.get("kind") == "stopped"), None)
        if failures:
            raise self._raised_failure(failures[0])
        if still_alive:
            raise RuntimeError("power sampler process could not be terminated")
        if not clean_shutdown:
            raise RuntimeError("power sampler process did not terminate cleanly")
        if self._process.exitcode != 0:
            raise RuntimeError(
                f"power sampler process exited during runtime with code {self._process.exitcode}"
            )
        if stopped is None:
            raise RuntimeError("power sampler process did not report orderly shutdown")
        self.sample_count = int(stopped["sample_count"])
        self.error_count = int(stopped["error_count"])

    def __enter__(self) -> ProcessPowerSampler:
        self.start()
        return self

    def __exit__(self, _kind, _value, _traceback) -> None:
        self.stop()


class NvmlProcessSampler(ProcessPowerSampler):
    """Production NVML sampler whose adapter exists only in its spawned child."""

    def __init__(
        self,
        expected_uuid: str,
        visible_device_count: int,
        path: str | Path,
        interval_ms: int = 100,
        execution_attempt: int = 1,
    ) -> None:
        super().__init__(
            _NvmlAdapterFactory(expected_uuid, visible_device_count),
            path,
            interval_ms,
            execution_attempt,
        )
