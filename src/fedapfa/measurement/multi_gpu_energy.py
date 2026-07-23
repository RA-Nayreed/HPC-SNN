"""Node-local telemetry validation, merging, and per-device energy accounting."""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from fedapfa.distributed.process_context import canonical_gpu_uuid
from fedapfa.utilities.serialization import atomic_write_json, atomic_write_text

from .energy import EnergyEstimate, integrate_energy
from .power import DeviceSample, _decode

NODE_TELEMETRY_SCHEMA_VERSION = 1
NODE_TELEMETRY_FIELDS = frozenset(
    {
        "schema_version",
        "monotonic_timestamp_ns",
        "utc_timestamp",
        "gpu_uuid_raw",
        "gpu_uuid",
        "node_identity",
        "power_watts",
        "gpu_utilization_percent",
        "memory_utilization_percent",
        "allocated_device_memory_bytes",
        "temperature_celsius",
        "graphics_clock_mhz",
        "memory_clock_mhz",
        "cumulative_device_energy_millijoules",
        "sampling_backend",
        "configured_interval_ms",
        "sampling_error_status",
        "execution_attempt",
        "slurm_allocation_identity",
    }
)


@dataclass(frozen=True)
class NodeTelemetrySample:
    schema_version: int
    monotonic_timestamp_ns: int
    utc_timestamp: str
    gpu_uuid_raw: str
    gpu_uuid: str
    node_identity: str
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
    execution_attempt: int
    slurm_allocation_identity: str

    def record(self) -> dict:
        return asdict(self)

    def device_sample(self) -> DeviceSample:
        return DeviceSample(
            self.monotonic_timestamp_ns,
            self.utc_timestamp,
            self.gpu_uuid,
            self.power_watts,
            self.gpu_utilization_percent,
            self.memory_utilization_percent,
            self.allocated_device_memory_bytes,
            self.temperature_celsius,
            self.graphics_clock_mhz,
            self.memory_clock_mhz,
            self.cumulative_device_energy_millijoules,
            self.sampling_backend,
            self.configured_interval_ms,
            self.sampling_error_status,
            self.execution_attempt,
        )


def _finite_optional(value, label: str, *, nonnegative: bool = True) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"telemetry {label} must be finite or null")
    if nonnegative and value < 0:
        raise ValueError(f"telemetry {label} must be nonnegative")


def _validate_sample(value: Mapping, path: Path, line_number: int) -> NodeTelemetrySample:
    if set(value) != NODE_TELEMETRY_FIELDS:
        missing = sorted(NODE_TELEMETRY_FIELDS - set(value))
        unexpected = sorted(set(value) - NODE_TELEMETRY_FIELDS)
        raise ValueError(f"{path}:{line_number} telemetry schema differs: missing={missing}, unexpected={unexpected}")
    try:
        sample = NodeTelemetrySample(**value)
        canonical = canonical_gpu_uuid(sample.gpu_uuid_raw)
        declared = canonical_gpu_uuid(sample.gpu_uuid)
        datetime.fromisoformat(sample.utc_timestamp.replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}:{line_number} has malformed telemetry identity or timestamp") from error
    if sample.schema_version != NODE_TELEMETRY_SCHEMA_VERSION:
        raise ValueError(f"{path}:{line_number} has an unsupported telemetry schema version")
    if declared != canonical or sample.gpu_uuid != canonical:
        raise ValueError(f"{path}:{line_number} canonical GPU UUID is not lowercase, hyphenated, and prefix-free")
    if not sample.node_identity or not sample.slurm_allocation_identity:
        raise ValueError(f"{path}:{line_number} lacks node or allocation identity")
    if sample.monotonic_timestamp_ns < 0 or sample.execution_attempt <= 0:
        raise ValueError(f"{path}:{line_number} has invalid monotonic or attempt identity")
    if sample.sampling_backend != "nvml" or sample.configured_interval_ms != 100:
        raise ValueError(f"{path}:{line_number} has incompatible sampling provenance")
    for field in (
        "power_watts",
        "gpu_utilization_percent",
        "memory_utilization_percent",
        "allocated_device_memory_bytes",
        "temperature_celsius",
        "graphics_clock_mhz",
        "memory_clock_mhz",
        "cumulative_device_energy_millijoules",
    ):
        _finite_optional(getattr(sample, field), field)
    for field in ("gpu_utilization_percent", "memory_utilization_percent"):
        value = getattr(sample, field)
        if value is not None and value > 100:
            raise ValueError(f"{path}:{line_number} has {field} above 100 percent")
    if sample.sampling_error_status is None and any(
        getattr(sample, field) is None
        for field in ("power_watts", "gpu_utilization_percent", "memory_utilization_percent")
    ):
        raise ValueError(f"{path}:{line_number} successful telemetry lacks required measurements")
    return sample


def read_node_telemetry(
    path: str | Path,
    *,
    expected_uuids: Sequence[str],
    expected_node_identity: str,
    execution_attempt: int,
    slurm_allocation_identity: str,
    reject_sampling_errors: bool = True,
) -> list[NodeTelemetrySample]:
    """Validate one node-owned JSONL file with exact device and attempt coverage."""

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"node telemetry file is missing: {source}")
    expected = tuple(canonical_gpu_uuid(value) for value in expected_uuids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected node GPU UUIDs must be nonempty and distinct")
    rows = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{source}:{line_number} contains an empty telemetry row")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source}:{line_number} is not valid JSON") from error
        if not isinstance(value, Mapping):
            raise ValueError(f"{source}:{line_number} telemetry row must be a mapping")
        rows.append(_validate_sample(value, source, line_number))
    if not rows:
        raise ValueError(f"node telemetry file contains no samples: {source}")
    if {row.node_identity for row in rows} != {expected_node_identity}:
        raise ValueError(f"node telemetry identity differs: {source}")
    if {row.execution_attempt for row in rows} != {execution_attempt}:
        raise ValueError(f"node telemetry mixes execution attempts: {source}")
    if {row.slurm_allocation_identity for row in rows} != {slurm_allocation_identity}:
        raise ValueError(f"node telemetry mixes Slurm allocations: {source}")
    observed = {row.gpu_uuid for row in rows}
    if observed != set(expected):
        raise ValueError(
            f"node telemetry GPU coverage differs: missing={sorted(set(expected) - observed)}, "
            f"unexpected={sorted(observed - set(expected))}"
        )
    if reject_sampling_errors and any(row.sampling_error_status is not None for row in rows):
        raise ValueError(f"node telemetry contains sampling-error rows: {source}")
    for gpu_uuid in expected:
        timestamps = [row.monotonic_timestamp_ns for row in rows if row.gpu_uuid == gpu_uuid]
        if len(timestamps) < 2:
            raise ValueError(f"node telemetry requires at least two samples for {gpu_uuid}")
        if any(right <= left for left, right in zip(timestamps, timestamps[1:], strict=False)):
            raise ValueError(f"node telemetry timestamps are duplicate or nonmonotonic for {gpu_uuid}")
    return rows


def merge_node_telemetry(
    node_files: Sequence[str | Path] | Mapping[str, str | Path],
    output_path: str | Path,
    *,
    expected_uuids_by_node: Mapping[str, Sequence[str]],
    execution_attempt: int,
    slurm_allocation_identity: str,
) -> list[NodeTelemetrySample]:
    """Validate node files, require global UUID coverage, and atomically merge them."""

    if isinstance(node_files, Mapping):
        if set(node_files) != set(expected_uuids_by_node):
            raise ValueError("node telemetry file identities differ from the expected nodes")
        node_paths = {str(node_identity): Path(path).resolve() for node_identity, path in node_files.items()}
    else:
        resolved_input = [Path(value).resolve() for value in node_files]
        if not len(set(resolved_input)) == len(resolved_input):
            raise ValueError("each node must own exactly one distinct telemetry file")
        node_paths = {}
        for path in resolved_input:
            try:
                first_row = next(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
                node_identity = str(json.loads(first_row)["node_identity"])
            except (StopIteration, KeyError, TypeError, json.JSONDecodeError, OSError) as error:
                raise ValueError(f"node telemetry identity cannot be resolved: {path}") from error
            if node_identity in node_paths:
                raise ValueError(f"node telemetry repeats node identity {node_identity}")
            node_paths[node_identity] = path
        if set(node_paths) != set(expected_uuids_by_node):
            raise ValueError("node telemetry file identities differ from the expected nodes")
    resolved_paths = list(node_paths.values())
    if len(resolved_paths) != len(expected_uuids_by_node) or len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("each node must own exactly one distinct telemetry file")
    if len(set(expected_uuids_by_node)) != len(expected_uuids_by_node):
        raise ValueError("node telemetry identities must be distinct")
    rows = []
    for node_identity, expected_uuids in sorted(expected_uuids_by_node.items()):
        path = node_paths[node_identity]
        rows.extend(
            read_node_telemetry(
                path,
                expected_uuids=expected_uuids,
                expected_node_identity=node_identity,
                execution_attempt=execution_attempt,
                slurm_allocation_identity=slurm_allocation_identity,
            )
        )
    expected_global = {canonical_gpu_uuid(value) for values in expected_uuids_by_node.values() for value in values}
    if len(expected_global) != sum(len(values) for values in expected_uuids_by_node.values()):
        raise ValueError("the global allocation repeats a physical GPU UUID")
    if {row.gpu_uuid for row in rows} != expected_global:
        raise ValueError("merged telemetry does not cover the exact global allocation")
    ordered = sorted(
        rows,
        key=lambda row: (row.monotonic_timestamp_ns, row.gpu_uuid, row.node_identity, row.utc_timestamp),
    )
    atomic_write_text(
        output_path,
        "".join(json.dumps(row.record(), sort_keys=True, allow_nan=False) + "\n" for row in ordered),
    )
    return ordered


def validate_client_interval_nonoverlap(intervals: Sequence[Mapping]) -> None:
    """Reject accepted client overlaps on one device while allowing cross-device concurrency."""

    by_device_attempt: dict[tuple[str, int], list[tuple[int, int, str]]] = {}
    for value in intervals:
        if value.get("category") != "client_training" or value.get("accepted") is not True:
            continue
        gpu_uuid = canonical_gpu_uuid(value["gpu_uuid"])
        start = int(value["start_ns"])
        end = int(value["end_ns"])
        if end <= start:
            raise ValueError("accepted client interval must have positive duration")
        by_device_attempt.setdefault((gpu_uuid, int(value["execution_attempt"])), []).append(
            (start, end, str(value.get("interval_id", "")))
        )
    for key, values in by_device_attempt.items():
        ordered = sorted(values)
        for left, right in zip(ordered, ordered[1:], strict=False):
            if right[0] < left[1]:
                raise ValueError(f"accepted client intervals overlap on physical device {key[0]}")


def integrate_physical_devices(
    samples: Sequence[NodeTelemetrySample],
    *,
    intervals_by_uuid: Mapping[str, tuple[int, int]],
    idle_baseline_watts_by_uuid: Mapping[str, float],
    maximum_gap_ms: int = 250,
) -> dict:
    """Integrate each physical GPU independently and sum only validated device totals."""

    observed = {row.gpu_uuid for row in samples}
    interval_uuids = {canonical_gpu_uuid(value) for value in intervals_by_uuid}
    baseline_uuids = {canonical_gpu_uuid(value) for value in idle_baseline_watts_by_uuid}
    if observed != interval_uuids or observed != baseline_uuids:
        raise ValueError("energy inputs must cover the exact same physical GPU UUID set")
    estimates: dict[str, EnergyEstimate] = {}
    for gpu_uuid in sorted(observed):
        start_ns, end_ns = intervals_by_uuid[gpu_uuid]
        estimates[gpu_uuid] = integrate_energy(
            [row.device_sample() for row in samples if row.gpu_uuid == gpu_uuid],
            int(start_ns),
            int(end_ns),
            float(idle_baseline_watts_by_uuid[gpu_uuid]),
            configured_interval_ms=100,
            maximum_gap_multiplier=maximum_gap_ms / 100,
        )
    return {
        "per_device": {gpu_uuid: estimate.record() for gpu_uuid, estimate in estimates.items()},
        "gross_energy_joules": sum(value.gross_energy_joules for value in estimates.values()),
        "idle_adjusted_energy_joules": sum(value.idle_adjusted_energy_joules for value in estimates.values()),
        "idle_baseline_contribution_joules": sum(
            value.idle_baseline_watts * value.coverage_seconds for value in estimates.values()
        ),
        "cumulative_energy_crosscheck_joules": (
            sum(float(value.cumulative_energy_crosscheck_joules) for value in estimates.values())
            if all(value.cumulative_energy_crosscheck_joules is not None for value in estimates.values())
            else None
        ),
        "device_count": len(estimates),
    }


def _allocation_gpu_uuid_checks(values, expected_count: int) -> tuple[bool, bool, tuple[str, ...]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return False, False, ()
    count_valid = len(values) == expected_count
    canonical = []
    validity = True
    for value in values:
        try:
            canonical.append(canonical_gpu_uuid(value))
        except ValueError:
            validity = False
    if len(canonical) != len(set(canonical)):
        validity = False
    return validity, count_valid, tuple(sorted(set(canonical)))


def validate_comparative_calibration(
    artifact: Mapping,
    *,
    requirements: Mapping,
    execution_gpu_uuids: Sequence[str],
    execution_commit: str,
) -> dict:
    """Validate allocation-local UUID evidence and cross-allocation measurement compatibility."""

    required_device_count = int(requirements["device_count"])
    calibration_uuid_validity, calibration_uuid_count, calibration_uuids = _allocation_gpu_uuid_checks(
        artifact.get("gpu_uuids"), required_device_count
    )
    execution_uuid_validity, execution_uuid_count, execution_uuids = _allocation_gpu_uuid_checks(
        execution_gpu_uuids, required_device_count
    )
    checks = {
        "passed": artifact.get("passed") is True,
        "paired_repetitions": int(artifact.get("paired_repetitions", 0)) >= int(requirements["paired_repetitions"]),
        "median_overhead": float(artifact.get("median_relative_overhead", math.inf))
        <= float(requirements["maximum_median_runtime_overhead_fraction"]),
        "sample_coverage": float(artifact.get("sample_coverage_fraction", 0.0))
        >= float(requirements["minimum_interval_coverage_fraction"]),
        "sampling_errors": artifact.get("sampling_errors") == [],
        "updates_identical": artifact.get("updates_numerically_identical") is True,
        "official_test_isolation": int(artifact.get("official_test_access_count", -1)) == 0,
        "node_count": artifact.get("node_count") == requirements["node_count"],
        "device_count": artifact.get("device_count") == requirements["device_count"],
        "process_count": artifact.get("process_count") == requirements["process_count"],
        "sampler_topology": artifact.get("sampler_topology") == requirements["sampler_topology"],
        "sampling_interval": artifact.get("sampling_interval_ms") == requirements["sampling_interval_ms"],
        "calibration_uuid_validity": calibration_uuid_validity,
        "calibration_uuid_count": calibration_uuid_count,
        "execution_uuid_validity": execution_uuid_validity,
        "execution_uuid_count": execution_uuid_count,
        "execution_commit": artifact.get("execution_commit") == execution_commit,
    }
    findings = [name for name, passed in checks.items() if not passed]
    if findings:
        raise ValueError(f"instrumentation calibration is incompatible: {sorted(set(findings))}")
    return {
        "compatible": True,
        "checks": checks,
        "calibration_allocation_gpu_uuids": list(calibration_uuids),
        "execution_allocation_gpu_uuids": list(execution_uuids),
    }


def _nvml_optional(binding, handle, function_name: str, *args):
    function = getattr(binding, function_name, None)
    if function is None:
        return None
    try:
        return function(handle, *args)
    except Exception:
        return None


def _node_sampler_process(
    expected_raw_uuids: tuple[str, ...],
    path: str,
    node_identity: str,
    execution_attempt: int,
    slurm_allocation_identity: str,
    stop_event,
    status_connection,
) -> None:
    binding = None
    handle = None
    sample_count = 0
    error_count = 0
    try:
        import pynvml as binding

        binding.nvmlInit()
        expected = {canonical_gpu_uuid(value): value for value in expected_raw_uuids}
        resolved = {}
        for index in range(int(binding.nvmlDeviceGetCount())):
            candidate = binding.nvmlDeviceGetHandleByIndex(index)
            raw = _decode(binding.nvmlDeviceGetUUID(candidate))
            canonical = canonical_gpu_uuid(raw)
            if canonical in expected:
                resolved[canonical] = (raw, candidate)
        if set(resolved) != set(expected) or len(resolved) != len(expected):
            raise RuntimeError("NVML devices do not exactly cover the node allocation")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        handle = output.open("x", encoding="utf-8")
        status_connection.send({"kind": "started", "pid": os.getpid()})
        interval_ns = 100_000_000
        next_ns = time.monotonic_ns()
        buffer = []
        while not stop_event.is_set():
            now_ns = time.monotonic_ns()
            utc = datetime.now(timezone.utc).isoformat()
            for canonical, (raw, device_handle) in sorted(resolved.items()):
                error_status = None
                try:
                    utilization = binding.nvmlDeviceGetUtilizationRates(device_handle)
                    memory = binding.nvmlDeviceGetMemoryInfo(device_handle)
                    power = float(binding.nvmlDeviceGetPowerUsage(device_handle)) / 1000.0
                    temperature = float(binding.nvmlDeviceGetTemperature(device_handle, binding.NVML_TEMPERATURE_GPU))
                    graphics = _nvml_optional(
                        binding, device_handle, "nvmlDeviceGetClockInfo", binding.NVML_CLOCK_GRAPHICS
                    )
                    memory_clock = _nvml_optional(
                        binding, device_handle, "nvmlDeviceGetClockInfo", binding.NVML_CLOCK_MEM
                    )
                    cumulative = _nvml_optional(binding, device_handle, "nvmlDeviceGetTotalEnergyConsumption")
                    gpu_utilization = float(utilization.gpu)
                    memory_utilization = float(utilization.memory)
                    memory_bytes = int(memory.used)
                except Exception as error:
                    error_status = f"{type(error).__name__}: {error}"
                    power = gpu_utilization = memory_utilization = memory_bytes = temperature = None
                    graphics = memory_clock = cumulative = None
                    error_count += 1
                row = NodeTelemetrySample(
                    NODE_TELEMETRY_SCHEMA_VERSION,
                    now_ns,
                    utc,
                    raw,
                    canonical,
                    node_identity,
                    power,
                    gpu_utilization,
                    memory_utilization,
                    memory_bytes,
                    temperature,
                    None if graphics is None else int(graphics),
                    None if memory_clock is None else int(memory_clock),
                    None if cumulative is None else int(cumulative),
                    "nvml",
                    100,
                    error_status,
                    execution_attempt,
                    slurm_allocation_identity,
                )
                buffer.append(json.dumps(row.record(), sort_keys=True, allow_nan=False) + "\n")
                sample_count += 1
            if len(buffer) >= 10 * len(resolved):
                handle.writelines(buffer)
                handle.flush()
                buffer.clear()
            next_ns += interval_ns
            stop_event.wait(max(0.0, (next_ns - time.monotonic_ns()) / 1_000_000_000))
        handle.writelines(buffer)
        handle.flush()
        os.fsync(handle.fileno())
        status_connection.send({"kind": "stopped", "sample_count": sample_count, "error_count": error_count})
    except BaseException as error:
        status_connection.send(
            {
                "kind": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            }
        )
    finally:
        if handle is not None:
            handle.close()
        if binding is not None:
            try:
                binding.nvmlShutdown()
            except Exception:
                pass
        status_connection.close()


class NodeNvmlProcessSampler:
    """Own one non-daemon sampler process and one telemetry writer for a physical node."""

    def __init__(
        self,
        expected_raw_uuids: Sequence[str],
        path: str | Path,
        *,
        node_identity: str,
        execution_attempt: int,
        slurm_allocation_identity: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        canonical = [canonical_gpu_uuid(value) for value in expected_raw_uuids]
        if not canonical or len(set(canonical)) != len(canonical):
            raise ValueError("node sampler requires distinct full-GPU UUIDs")
        self.expected_raw_uuids = tuple(expected_raw_uuids)
        self.path = Path(path)
        self.node_identity = node_identity
        self.execution_attempt = execution_attempt
        self.slurm_allocation_identity = slurm_allocation_identity
        self.timeout_seconds = timeout_seconds
        self.sample_count = 0
        self.error_count = 0
        self._context = multiprocessing.get_context("spawn")
        self._stop = None
        self._process = None
        self._receiver = None

    @property
    def process_is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def _terminate(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._process is not None:
            self._process.join(timeout=self.timeout_seconds)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=self.timeout_seconds)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=self.timeout_seconds)

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("node sampler has already started")
        self._stop = self._context.Event()
        self._receiver, sender = self._context.Pipe(duplex=False)
        self._process = self._context.Process(
            target=_node_sampler_process,
            args=(
                self.expected_raw_uuids,
                str(self.path),
                self.node_identity,
                self.execution_attempt,
                self.slurm_allocation_identity,
                self._stop,
                sender,
            ),
            name="fedapfa-node-nvml-sampler",
            daemon=False,
        )
        self._process.start()
        sender.close()
        deadline = time.monotonic() + self.timeout_seconds
        try:
            while time.monotonic() < deadline:
                if self._receiver.poll(0.05):
                    status = self._receiver.recv()
                    if status.get("kind") == "started":
                        return
                    raise RuntimeError(
                        f"node sampler startup failed: {status.get('error_type')}: {status.get('error_message')}"
                    )
                if self._process.exitcode is not None:
                    raise RuntimeError(f"node sampler exited during startup with code {self._process.exitcode}")
            raise RuntimeError("node sampler startup timed out")
        except BaseException:
            self._terminate()
            raise

    def stop(self) -> None:
        if self._process is None or self._stop is None or self._receiver is None:
            raise RuntimeError("node sampler was not started")
        self._stop.set()
        self._process.join(timeout=self.timeout_seconds)
        clean_shutdown = not self._process.is_alive()
        if not clean_shutdown:
            self._terminate()
        still_alive = self._process.is_alive()
        statuses = []
        try:
            while self._receiver.poll():
                statuses.append(self._receiver.recv())
        except EOFError:
            pass
        finally:
            receiver = self._receiver
            self._receiver = None
            receiver.close()
        failed = next((value for value in statuses if value.get("kind") == "failed"), None)
        stopped = next((value for value in statuses if value.get("kind") == "stopped"), None)
        if failed is not None:
            raise RuntimeError(f"node sampler failed: {failed.get('error_type')}: {failed.get('error_message')}")
        if still_alive:
            raise RuntimeError("node sampler process could not be terminated")
        if not clean_shutdown:
            raise RuntimeError("node sampler did not terminate cleanly")
        if self._process.exitcode != 0:
            raise RuntimeError(f"node sampler exited during runtime with code {self._process.exitcode}")
        if stopped is None:
            raise RuntimeError("node sampler lacks an orderly shutdown record")
        self.sample_count = int(stopped["sample_count"])
        self.error_count = int(stopped["error_count"])

    def abort(self) -> None:
        """Stop the sampler without requiring an orderly status exchange."""

        self._terminate()
        if self._receiver is not None:
            self._receiver.close()
            self._receiver = None


def write_energy_summary(path: str | Path, value: Mapping) -> None:
    """Write validated energy evidence atomically."""

    atomic_write_json(path, dict(value))
