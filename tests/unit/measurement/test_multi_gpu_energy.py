from __future__ import annotations

import hashlib
import json
import multiprocessing
import signal
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fedapfa.measurement import comparative_runtime
from fedapfa.measurement.client_interval import IntervalRecorder
from fedapfa.measurement.comparative_runtime import ComparativeMeasurementSession
from fedapfa.measurement.multi_gpu_energy import (
    NodeNvmlProcessSampler,
    NodeTelemetrySample,
    integrate_physical_devices,
    merge_node_telemetry,
    read_node_telemetry,
    validate_client_interval_nonoverlap,
    validate_comparative_calibration,
)

UUIDS = (
    "11111111-1111-1111-1111-111111111111",
    "22222222-2222-2222-2222-222222222222",
    "33333333-3333-3333-3333-333333333333",
    "44444444-4444-4444-4444-444444444444",
)


def _sample(uuid: str, timestamp: int, *, node: str = "node-a", power: float = 100.0) -> NodeTelemetrySample:
    return NodeTelemetrySample(
        1,
        timestamp,
        "2026-07-21T00:00:00+00:00",
        f"GPU-{uuid.upper()}",
        uuid,
        node,
        power,
        50.0,
        25.0,
        1000,
        40.0,
        1200,
        900,
        timestamp // 1_000_000,
        "nvml",
        100,
        None,
        1,
        "100_0:101",
    )


def _write(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(value.record(), sort_keys=True) + "\n" for value in rows),
        encoding="utf-8",
    )


class _NodeSamplerStopEvent:
    def __init__(self) -> None:
        self.set_count = 0

    def set(self) -> None:
        self.set_count += 1


class _NodeSamplerProcess:
    def __init__(self, *, exitcode: int, alive: bool = False) -> None:
        self.exitcode = exitcode
        self._alive = alive
        self.join_count = 0
        self.terminate_count = 0
        self.kill_count = 0

    def join(self, timeout: float) -> None:
        assert timeout > 0
        self.join_count += 1

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminate_count += 1
        self._alive = False
        self.exitcode = -15

    def kill(self) -> None:
        self.kill_count += 1
        self._alive = False
        self.exitcode = -9


def _node_sampler_for_stop(tmp_path: Path, *, exitcode: int = 0, alive: bool = False):
    sampler = NodeNvmlProcessSampler(
        UUIDS[:1],
        tmp_path / "node.jsonl",
        node_identity="node-a",
        execution_attempt=1,
        slurm_allocation_identity="100_0:101",
        timeout_seconds=0.1,
    )
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = _NodeSamplerProcess(exitcode=exitcode, alive=alive)
    stop_event = _NodeSamplerStopEvent()
    sampler._receiver = receiver
    sampler._process = process
    sampler._stop = stop_event
    return sampler, receiver, sender, process, stop_event


def test_node_sampler_stop_accepts_stopped_status_followed_by_pipe_eof(tmp_path: Path) -> None:
    sampler, receiver, sender, process, stop_event = _node_sampler_for_stop(tmp_path)
    sender.send({"kind": "stopped", "sample_count": 23, "error_count": 2})
    sender.close()
    assert receiver.poll(1.0)

    sampler.stop()

    assert sampler.sample_count == 23
    assert sampler.error_count == 2
    assert not process.is_alive()
    assert stop_event.set_count == 1
    assert receiver.closed
    assert sampler._receiver is None


def test_node_sampler_stop_rejects_pipe_eof_without_stopped_status(tmp_path: Path) -> None:
    sampler, receiver, sender, process, _ = _node_sampler_for_stop(tmp_path)
    sender.close()
    assert receiver.poll(1.0)

    with pytest.raises(RuntimeError, match="lacks an orderly shutdown record"):
        sampler.stop()

    assert not process.is_alive()
    assert receiver.closed
    assert sampler._receiver is None


def test_node_sampler_stop_preserves_failed_status_before_pipe_eof(tmp_path: Path) -> None:
    sampler, receiver, sender, process, _ = _node_sampler_for_stop(tmp_path)
    sender.send(
        {
            "kind": "failed",
            "error_type": "SensorFailure",
            "error_message": "telemetry write failed",
        }
    )
    sender.close()
    assert receiver.poll(1.0)

    with pytest.raises(RuntimeError, match="node sampler failed: SensorFailure: telemetry write failed"):
        sampler.stop()

    assert not process.is_alive()
    assert receiver.closed
    assert sampler._receiver is None


def test_node_sampler_stop_rejects_nonzero_exit_after_stopped_status_and_eof(tmp_path: Path) -> None:
    sampler, receiver, sender, process, _ = _node_sampler_for_stop(tmp_path, exitcode=7)
    sender.send({"kind": "stopped", "sample_count": 5, "error_count": 0})
    sender.close()
    assert receiver.poll(1.0)

    with pytest.raises(RuntimeError, match="exited during runtime with code 7"):
        sampler.stop()

    assert not process.is_alive()
    assert receiver.closed
    assert sampler._receiver is None


def test_node_sampler_stop_terminates_unclean_child_and_rejects_shutdown(tmp_path: Path) -> None:
    sampler, receiver, sender, process, stop_event = _node_sampler_for_stop(tmp_path, alive=True)
    sender.close()
    assert receiver.poll(1.0)

    with pytest.raises(RuntimeError, match="did not terminate cleanly"):
        sampler.stop()

    assert not process.is_alive()
    assert process.terminate_count == 1
    assert process.kill_count == 0
    assert stop_event.set_count == 2
    assert receiver.closed
    assert sampler._receiver is None


@pytest.mark.parametrize("count", [1, 2, 4])
def test_node_file_exact_device_coverage(tmp_path: Path, count: int) -> None:
    path = tmp_path / "node.jsonl"
    rows = [_sample(uuid, timestamp) for timestamp in (0, 100_000_000, 200_000_000) for uuid in UUIDS[:count]]
    _write(path, rows)
    observed = read_node_telemetry(
        path,
        expected_uuids=UUIDS[:count],
        expected_node_identity="node-a",
        execution_attempt=1,
        slurm_allocation_identity="100_0:101",
    )
    assert {value.gpu_uuid for value in observed} == set(UUIDS[:count])


def test_two_node_merge_is_deterministic_and_atomic(tmp_path: Path) -> None:
    first = tmp_path / "node-0.jsonl"
    second = tmp_path / "node-1.jsonl"
    _write(first, [_sample(uuid, timestamp) for timestamp in (0, 100_000_000) for uuid in UUIDS[:2]])
    _write(
        second,
        [
            replace(_sample(uuid, timestamp), node_identity="node-b")
            for timestamp in (0, 100_000_000)
            for uuid in UUIDS[2:]
        ],
    )
    output = tmp_path / "merged.jsonl"
    rows = merge_node_telemetry(
        [second, first],
        output,
        expected_uuids_by_node={"node-a": UUIDS[:2], "node-b": UUIDS[2:]},
        execution_attempt=1,
        slurm_allocation_identity="100_0:101",
    )
    assert len(rows) == 8
    first_bytes = output.read_bytes()
    merge_node_telemetry(
        [first, second],
        output,
        expected_uuids_by_node={"node-a": UUIDS[:2], "node-b": UUIDS[2:]},
        execution_attempt=1,
        slurm_allocation_identity="100_0:101",
    )
    assert output.read_bytes() == first_bytes


def test_node_merge_rejects_shared_writer_path(tmp_path: Path) -> None:
    path = tmp_path / "node.jsonl"
    _write(path, [_sample(UUIDS[0], value) for value in (0, 100_000_000)])
    with pytest.raises(ValueError, match="distinct telemetry file"):
        merge_node_telemetry(
            [path, path],
            tmp_path / "merged.jsonl",
            expected_uuids_by_node={"node-a": UUIDS[:1], "node-b": UUIDS[1:2]},
            execution_attempt=1,
            slurm_allocation_identity="100_0:101",
        )


def test_per_device_energy_integrates_then_sums_once() -> None:
    rows = [
        _sample(uuid, timestamp, power=power)
        for uuid, power in ((UUIDS[0], 100.0), (UUIDS[1], 200.0))
        for timestamp in (0, 100_000_000, 200_000_000)
    ]
    result = integrate_physical_devices(
        rows,
        intervals_by_uuid={uuid: (0, 200_000_000) for uuid in UUIDS[:2]},
        idle_baseline_watts_by_uuid={UUIDS[0]: 10.0, UUIDS[1]: 20.0},
    )
    assert result["gross_energy_joules"] == pytest.approx(60.0)
    assert result["idle_adjusted_energy_joules"] == pytest.approx(54.0)
    assert result["device_count"] == 2


def test_same_device_client_overlap_rejected_cross_device_allowed() -> None:
    base = {
        "category": "client_training",
        "accepted": True,
        "execution_attempt": 1,
        "start_ns": 0,
        "end_ns": 10,
        "interval_id": "a",
        "gpu_uuid": UUIDS[0],
    }
    validate_client_interval_nonoverlap([base, {**base, "interval_id": "b", "gpu_uuid": UUIDS[1]}])
    with pytest.raises(ValueError, match="overlap"):
        validate_client_interval_nonoverlap([base, {**base, "interval_id": "b", "start_ns": 9, "end_ns": 20}])


def test_sampling_error_and_attempt_mixing_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "node.jsonl"
    rows = [_sample(UUIDS[0], 0), replace(_sample(UUIDS[0], 100_000_000), sampling_error_status="lost")]
    _write(path, rows)
    with pytest.raises(ValueError, match="sampling-error"):
        read_node_telemetry(
            path,
            expected_uuids=UUIDS[:1],
            expected_node_identity="node-a",
            execution_attempt=1,
            slurm_allocation_identity="100_0:101",
        )


def test_gap_rejection_and_boundary_interpolation_are_per_device() -> None:
    with pytest.raises(ValueError, match="gap exceeds"):
        integrate_physical_devices(
            [_sample(UUIDS[0], 0), _sample(UUIDS[0], 300_000_000)],
            intervals_by_uuid={UUIDS[0]: (0, 300_000_000)},
            idle_baseline_watts_by_uuid={UUIDS[0]: 0.0},
            maximum_gap_ms=250,
        )
    rows = [
        _sample(UUIDS[0], 0, power=0.0),
        _sample(UUIDS[0], 100_000_000, power=100.0),
        _sample(UUIDS[0], 200_000_000, power=200.0),
    ]
    result = integrate_physical_devices(
        rows,
        intervals_by_uuid={UUIDS[0]: (50_000_000, 150_000_000)},
        idle_baseline_watts_by_uuid={UUIDS[0]: 25.0},
    )
    assert result["gross_energy_joules"] == pytest.approx(10.0)
    assert result["idle_adjusted_energy_joules"] == pytest.approx(7.5)
    assert result["cumulative_energy_crosscheck_joules"] is not None


def test_interrupted_client_interval_is_not_an_accepted_overlap() -> None:
    interrupted = {
        "category": "client_training",
        "accepted": False,
        "execution_attempt": 1,
        "start_ns": 0,
        "end_ns": 20,
        "interval_id": "interrupted",
        "gpu_uuid": UUIDS[0],
    }
    accepted = {**interrupted, "accepted": True, "start_ns": 5, "end_ns": 10}
    validate_client_interval_nonoverlap([interrupted, accepted])


def test_comparative_session_strict_lifo_guard_and_reverse_abort_are_retained(tmp_path: Path) -> None:
    session = ComparativeMeasurementSession.__new__(ComparativeMeasurementSession)
    session._started = True
    session._open_scopes = []
    session.execution_attempt = 1
    session.gpu_uuid = UUIDS[0]
    interval_path = tmp_path / "execution_intervals.jsonl"
    session.intervals = IntervalRecorder(interval_path)

    treatment_token = session.begin("complete_treatment")
    round_token = session.begin("communication_round", {"round_number": 1})

    with pytest.raises(RuntimeError, match="measurement interval nesting is incompatible"):
        session.end(treatment_token)
    assert [token for _, token in session._open_scopes] == [treatment_token, round_token]

    session._abort_scopes()

    records = [json.loads(line) for line in interval_path.read_text(encoding="utf-8").splitlines()]
    assert [value["category"] for value in records] == [
        "communication_round",
        "complete_treatment",
    ]
    assert all(value["accepted"] is False for value in records)
    assert all(value["exclusion_reason"] == "RuntimeError" for value in records)
    assert session._open_scopes == []
    with pytest.raises(RuntimeError, match="measurement interval nesting is incompatible"):
        session.end(round_token)


def _calibration_requirements() -> dict:
    return {
        "paired_repetitions": 10,
        "maximum_median_runtime_overhead_fraction": 0.02,
        "minimum_interval_coverage_fraction": 0.9,
        "node_count": 1,
        "device_count": 2,
        "process_count": 2,
        "sampler_topology": "1_node_2_device_node_local",
        "sampling_interval_ms": 100,
    }


def _calibration_artifact() -> dict:
    return {
        "passed": True,
        "paired_repetitions": 10,
        "median_relative_overhead": 0.01,
        "sample_coverage_fraction": 0.9,
        "sampling_errors": [],
        "updates_numerically_identical": True,
        "official_test_access_count": 0,
        "node_count": 1,
        "device_count": 2,
        "process_count": 2,
        "sampler_topology": "1_node_2_device_node_local",
        "sampling_interval_ms": 100,
        "gpu_uuids": list(UUIDS[:2]),
        "execution_commit": "abc",
    }


def test_cross_allocation_calibration_accepts_distinct_canonical_uuid_sets() -> None:
    artifact = _calibration_artifact()
    artifact["gpu_uuids"] = [f" GPU-{UUIDS[0].upper()} ", UUIDS[1].upper()]
    result = validate_comparative_calibration(
        artifact,
        requirements=_calibration_requirements(),
        execution_gpu_uuids=[f" gpu-{UUIDS[2].upper()} ", UUIDS[3].upper()],
        execution_commit="abc",
    )
    assert result["compatible"]
    assert result["calibration_allocation_gpu_uuids"] == list(UUIDS[:2])
    assert result["execution_allocation_gpu_uuids"] == list(UUIDS[2:])
    assert result["checks"]["calibration_uuid_validity"]
    assert result["checks"]["calibration_uuid_count"]
    assert result["checks"]["execution_uuid_validity"]
    assert result["checks"]["execution_uuid_count"]
    assert "gpu_uuid_coverage" not in result["checks"]


@pytest.mark.parametrize(
    ("values", "failed_check"),
    [
        ([UUIDS[0], "malformed"], "calibration_uuid_validity"),
        ([UUIDS[0], " "], "calibration_uuid_validity"),
        ([UUIDS[0], f"MIG-{UUIDS[1]}"], "calibration_uuid_validity"),
        ([UUIDS[0], f"GPU-{UUIDS[0].upper()}"], "calibration_uuid_validity"),
        ([UUIDS[0]], "calibration_uuid_count"),
        ([UUIDS[0], UUIDS[1], UUIDS[2]], "calibration_uuid_count"),
    ],
)
def test_invalid_calibration_allocation_uuid_evidence_is_rejected(values, failed_check: str) -> None:
    artifact = _calibration_artifact()
    artifact["gpu_uuids"] = values
    with pytest.raises(ValueError, match=failed_check):
        validate_comparative_calibration(
            artifact,
            requirements=_calibration_requirements(),
            execution_gpu_uuids=UUIDS[2:],
            execution_commit="abc",
        )


@pytest.mark.parametrize(
    ("values", "failed_check"),
    [
        ([UUIDS[2], "malformed"], "execution_uuid_validity"),
        ([UUIDS[2], ""], "execution_uuid_validity"),
        ([UUIDS[2], f"MIG-{UUIDS[3]}"], "execution_uuid_validity"),
        ([UUIDS[2], f"GPU-{UUIDS[2].upper()}"], "execution_uuid_validity"),
        ([UUIDS[2]], "execution_uuid_count"),
        ([UUIDS[1], UUIDS[2], UUIDS[3]], "execution_uuid_count"),
    ],
)
def test_invalid_execution_allocation_uuid_evidence_is_rejected(values, failed_check: str) -> None:
    with pytest.raises(ValueError, match=failed_check):
        validate_comparative_calibration(
            _calibration_artifact(),
            requirements=_calibration_requirements(),
            execution_gpu_uuids=values,
            execution_commit="abc",
        )


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    [
        ("passed", False, "passed"),
        ("paired_repetitions", 9, "paired_repetitions"),
        ("median_relative_overhead", 0.03, "median_overhead"),
        ("sample_coverage_fraction", 0.89, "sample_coverage"),
        ("sampling_errors", ["sample failed"], "sampling_errors"),
        ("updates_numerically_identical", False, "updates_identical"),
        ("official_test_access_count", 1, "official_test_isolation"),
        ("node_count", 2, "node_count"),
        ("device_count", 1, "device_count"),
        ("process_count", 1, "process_count"),
        ("sampler_topology", "incompatible", "sampler_topology"),
        ("sampling_interval_ms", 200, "sampling_interval"),
        ("execution_commit", "def", "execution_commit"),
    ],
)
def test_non_uuid_calibration_compatibility_checks_remain_strict(field: str, value, failed_check: str) -> None:
    artifact = _calibration_artifact()
    artifact[field] = value
    with pytest.raises(ValueError, match=failed_check):
        validate_comparative_calibration(
            artifact,
            requirements=_calibration_requirements(),
            execution_gpu_uuids=UUIDS[2:],
            execution_commit="abc",
        )


class _StartableSampler:
    process_is_alive = True

    def __init__(self, *args, **kwargs) -> None:
        self.started = False

    def start(self) -> None:
        self.started = True


def test_comparative_session_records_both_allocation_uuid_sets(tmp_path: Path, monkeypatch) -> None:
    artifact = _calibration_artifact()
    artifact["gpu_uuids"] = [f"GPU-{UUIDS[0].upper()}", UUIDS[1]]
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(artifact), encoding="utf-8")
    calibration_sha256 = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    config = {
        "energy_measurement": {"idle_before_seconds": 0.0},
        "model": {"hidden_dims": [1, 1]},
        "frozen_model_diagnostics": {
            "runtime": {"model_path": "runtime.json", "model_sha256": "runtime", "feature_order": []},
            "gross_energy": {"model_path": "gross.json", "model_sha256": "gross", "feature_order": []},
        },
        "calibration_requirements": _calibration_requirements(),
        "instrumentation_calibration_identity": {"sha256": calibration_sha256},
    }
    execution_values = [f"GPU-{UUIDS[2].upper()}", f"GPU-{UUIDS[3].upper()}"]
    context = SimpleNamespace(
        device=torch.device("cuda:0"),
        control_device=torch.device("cuda:0"),
        client_processes_per_device=1,
        gpu_uuid=execution_values[0],
        gpu_uuid_raw=execution_values[0],
        world_size=2,
        rank=0,
        node_rank=0,
        local_rank=0,
        node_count=1,
        physical_device_count=2,
        host="execution-node",
        is_coordinator=True,
    )

    def all_gather(values, value):
        if isinstance(value, dict) and "gpu_uuid" in value:
            values[:] = [
                value,
                {
                    "rank": 1,
                    "node_rank": 0,
                    "host": "execution-node",
                    "gpu_uuid": UUIDS[3],
                    "gpu_uuid_raw": execution_values[1],
                },
            ]
        else:
            values[:] = [None, None]

    def frozen_model(path, expected_sha256, feature_order):
        target = "gross_energy_joules" if "gross" in str(path) else "client_wall_time_seconds"
        return {"target": target}

    monkeypatch.setenv("SLURM_JOB_ID", "200")
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    monkeypatch.setattr(comparative_runtime.dist, "all_gather_object", all_gather)
    monkeypatch.setattr(comparative_runtime.dist, "broadcast_object_list", lambda values, src, device: None)
    monkeypatch.setattr(comparative_runtime.dist, "barrier", lambda: None)
    monkeypatch.setattr(comparative_runtime, "git_metadata", lambda: {"commit": "abc"})
    monkeypatch.setattr(comparative_runtime, "_load_frozen_model", frozen_model)
    monkeypatch.setattr(comparative_runtime, "NodeNvmlProcessSampler", _StartableSampler)
    model = SimpleNamespace(state_dict=lambda: {"weight": torch.tensor([1.0])})

    session = ComparativeMeasurementSession(
        config, tmp_path / "run", SimpleNamespace(), model, context, calibration_path
    )
    session.start()

    provenance = json.loads(
        (session.attempt_dir / "calibration_reference.json").read_text(encoding="utf-8")
    )
    assert provenance["calibration_allocation_gpu_uuids"] == list(UUIDS[:2])
    assert provenance["execution_allocation_gpu_uuids"] == list(UUIDS[2:])
    assert provenance["compatibility_checks"]["calibration_uuid_validity"]
    assert provenance["compatibility_checks"]["execution_uuid_validity"]


class _FailingSampler:
    def __init__(self) -> None:
        self.aborted = False

    @property
    def process_is_alive(self) -> bool:
        return not self.aborted

    def start(self) -> None:
        raise RuntimeError("sampler boom")

    def abort(self) -> None:
        self.aborted = True


def test_session_start_exchanges_sampler_failure_before_raising(tmp_path: Path, monkeypatch) -> None:
    session = ComparativeMeasurementSession.__new__(ComparativeMeasurementSession)
    session.rank_dir = tmp_path / "rank_0"
    session.interval_path = session.rank_dir / "intervals.jsonl"
    session.provisional_path = session.rank_dir / "provisional.jsonl"
    session.excluded_interval_path = session.rank_dir / "excluded.jsonl"
    session.context = SimpleNamespace(
        is_coordinator=False,
        world_size=1,
        local_rank=0,
        rank=0,
        node_rank=0,
    )
    sampler = _FailingSampler()
    session.node_sampler = sampler
    exchanged = []
    monkeypatch.setattr(comparative_runtime.dist, "barrier", lambda: None)

    def all_gather(values, value):
        exchanged.append(value)
        values[:] = [value]

    monkeypatch.setattr(comparative_runtime.dist, "all_gather_object", all_gather)
    with pytest.raises(RuntimeError, match="sampler boom"):
        session.start()
    assert sampler.aborted
    assert exchanged[0]["error_type"] == "RuntimeError"


class _StoppingSampler:
    def __init__(self) -> None:
        self.stopped = False

    @property
    def process_is_alive(self) -> bool:
        return False

    def stop(self) -> None:
        self.stopped = True


class _Clock:
    def __init__(self) -> None:
        self.value = 0

    def now_ns(self) -> int:
        self.value += 1
        return self.value


def test_session_stop_broadcasts_finalization_failure_after_stopping_sampler(monkeypatch) -> None:
    session = ComparativeMeasurementSession.__new__(ComparativeMeasurementSession)
    session._started = True
    session._abort_scopes = lambda: None
    session.context = SimpleNamespace(
        is_coordinator=True,
        world_size=1,
        local_rank=0,
        rank=0,
        node_rank=0,
        host="node-a",
        control_device=None,
    )
    session.measurement = {"idle_after_seconds": 0.0}
    session.clock = _Clock()
    session.idle_record = {"node_rank": 0, "node_identity": "node-a"}
    sampler = _StoppingSampler()
    session.node_sampler = sampler
    monkeypatch.setattr(comparative_runtime.dist, "barrier", lambda: None)

    def all_gather(values, value):
        values[:] = [value]

    monkeypatch.setattr(comparative_runtime.dist, "all_gather_object", all_gather)
    monkeypatch.setattr(comparative_runtime.dist, "broadcast_object_list", lambda values, src, device: None)
    session._finalize_attempt = lambda completed, idle: (_ for _ in ()).throw(RuntimeError("finalize boom"))
    with pytest.raises(RuntimeError, match="finalize boom"):
        session.stop(True)
    assert sampler.stopped


class _AbortSampler:
    def __init__(self) -> None:
        self.stopped = False
        self.aborted = False

    @property
    def process_is_alive(self) -> bool:
        return not self.stopped

    def stop(self) -> None:
        self.stopped = True

    def abort(self) -> None:
        self.aborted = True


def test_session_abort_flushes_recoverable_failure_evidence(tmp_path: Path) -> None:
    session = ComparativeMeasurementSession.__new__(ComparativeMeasurementSession)
    session._started = True
    session._abort_scopes = lambda: None
    session.attempt_dir = tmp_path / "measurement_attempts" / "attempt_1"
    session.interval_path = session.attempt_dir / "rank_0" / "execution_intervals.jsonl"
    session.interval_path.parent.mkdir(parents=True)
    interval = {
        "category": "complete_treatment",
        "start_ns": 10,
        "end_ns": 20,
        "accepted": False,
    }
    session.interval_path.write_text(json.dumps(interval) + "\n", encoding="utf-8")
    session.context = SimpleNamespace(
        rank=0,
        node_rank=0,
        local_rank=0,
        world_size=1,
        host="node-a",
    )
    session.execution_attempt = 1
    session.gpu_uuid = UUIDS[0]
    session.allocation_identity = "100_0:101"
    session.idle_record = {
        "node_rank": 0,
        "node_identity": "node-a",
        "idle_before": {"start_ns": 0, "end_ns": 9},
    }
    sampler = _AbortSampler()
    session.node_sampler = sampler

    session.abort(RuntimeError("preempted"))

    evidence = json.loads((session.attempt_dir / "rank_0_failure.json").read_text(encoding="utf-8"))
    assert sampler.stopped
    assert evidence["host"] == "node-a"
    assert evidence["idle_record"] == session.idle_record
    assert evidence["complete_treatment_interval"] == interval
    assert evidence["sampler_shutdown_error"] is None
    assert not session._started


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_signal_abort_unwinds_scopes_and_stops_sampler_child(tmp_path: Path, signum: signal.Signals) -> None:
    session = ComparativeMeasurementSession.__new__(ComparativeMeasurementSession)
    session._started = True
    session._open_scopes = []
    session.attempt_dir = tmp_path / "measurement_attempts" / "attempt_1"
    session.interval_path = session.attempt_dir / "rank_0" / "execution_intervals.jsonl"
    session.interval_path.parent.mkdir(parents=True)
    session.intervals = IntervalRecorder(session.interval_path)
    session.context = SimpleNamespace(
        rank=0,
        node_rank=0,
        local_rank=0,
        world_size=1,
        host="node-a",
    )
    session.execution_attempt = 1
    session.gpu_uuid = UUIDS[0]
    session.allocation_identity = "100_0:101"
    session.idle_record = {
        "node_rank": 0,
        "node_identity": "node-a",
        "idle_before": {"start_ns": 0, "end_ns": 1},
    }
    sampler = _AbortSampler()
    session.node_sampler = sampler
    session.begin("complete_treatment")
    session.begin("communication_round", {"round_number": 1})

    session.abort(SystemExit(128 + signum))

    intervals = [json.loads(line) for line in session.interval_path.read_text(encoding="utf-8").splitlines()]
    evidence = json.loads((session.attempt_dir / "rank_0_failure.json").read_text(encoding="utf-8"))
    assert [value["category"] for value in intervals] == ["communication_round", "complete_treatment"]
    assert all(value["accepted"] is False for value in intervals)
    assert session._open_scopes == []
    assert sampler.stopped and not sampler.aborted
    assert evidence["error_type"] == "SystemExit"
    assert evidence["error_message"] == str(128 + signum)
    assert not session._started


def test_interrupted_attempt_recovery_is_attempt_local_and_restores_state(tmp_path: Path) -> None:
    attempts = tmp_path / "measurement_attempts"
    interrupted = attempts / "attempt_1"
    interrupted.mkdir(parents=True)
    complete_interval = {
        "category": "complete_treatment",
        "start_ns": 10,
        "end_ns": 20,
        "accepted": False,
    }
    failure = {
        "execution_attempt": 1,
        "rank": 0,
        "node_rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "host": "old-node",
        "gpu_uuid": UUIDS[0],
        "slurm_allocation_identity": "100_0:101",
        "idle_record": {
            "node_rank": 0,
            "node_identity": "old-node",
            "idle_before": {"start_ns": 0, "end_ns": 9},
        },
        "complete_treatment_interval": complete_interval,
    }
    (interrupted / "rank_0_failure.json").write_text(json.dumps(failure), encoding="utf-8")
    session = ComparativeMeasurementSession.__new__(ComparativeMeasurementSession)
    session.run_dir = tmp_path
    session.attempt_dir = attempts / "attempt_2"
    session.execution_attempt = 2
    session.allocation_identity = "200_0:201"
    session.context = SimpleNamespace(world_size=1, node_count=1, physical_device_count=1)
    observed = {}

    def finalize(completed, idle_records, **kwargs):
        observed.update(
            {
                "completed": completed,
                "idle_records": idle_records,
                "kwargs": kwargs,
                "attempt_dir": session.attempt_dir,
                "execution_attempt": session.execution_attempt,
                "allocation_identity": session.allocation_identity,
            }
        )

    session._finalize_attempt = finalize
    assert session._recover_interrupted_attempts() == []
    assert observed["attempt_dir"] == interrupted
    assert observed["execution_attempt"] == 1
    assert observed["allocation_identity"] == "100_0:101"
    assert observed["kwargs"]["node_mapping"] == {"old-node": [UUIDS[0]]}
    assert observed["kwargs"]["node_files"] == {"old-node": interrupted / "node_0_samples.jsonl"}
    assert observed["kwargs"]["allow_missing_idle_after"]
    assert session.attempt_dir == attempts / "attempt_2"
    assert session.execution_attempt == 2
    assert session.allocation_identity == "200_0:201"
