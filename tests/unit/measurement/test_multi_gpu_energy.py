from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fedapfa.measurement import comparative_runtime
from fedapfa.measurement.comparative_runtime import ComparativeMeasurementSession
from fedapfa.measurement.multi_gpu_energy import (
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


def test_topology_calibration_compatibility_is_exact() -> None:
    requirements = {
        "paired_repetitions": 10,
        "maximum_median_runtime_overhead_fraction": 0.02,
        "minimum_interval_coverage_fraction": 0.9,
        "node_count": 1,
        "device_count": 2,
        "process_count": 2,
        "sampler_topology": "1_node_2_device_node_local",
        "sampling_interval_ms": 100,
    }
    artifact = {
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
    assert validate_comparative_calibration(
        artifact,
        requirements=requirements,
        expected_gpu_uuids=UUIDS[:2],
        execution_commit="abc",
    )["compatible"]
    incompatible = {**artifact, "device_count": 1}
    with pytest.raises(ValueError, match="device_count"):
        validate_comparative_calibration(
            incompatible,
            requirements=requirements,
            expected_gpu_uuids=UUIDS[:2],
            execution_commit="abc",
        )


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
