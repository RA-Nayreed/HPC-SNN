from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from fedapfa.measurement.client_interval import (
    ClientIntervalIdentity,
    IntervalRecord,
    IntervalRecorder,
)
from fedapfa.measurement.comparative_runtime import ComparativeMeasurementSession
from fedapfa.measurement.energy import integrate_energy
from fedapfa.measurement.multi_gpu_energy import NodeTelemetrySample

UUIDS = (
    "11111111-1111-1111-1111-111111111111",
    "22222222-2222-2222-2222-222222222222",
    "33333333-3333-3333-3333-333333333333",
    "44444444-4444-4444-4444-444444444444",
)
OFFSET_NS = 69_000 * 1_000_000_000


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True, allow_nan=False) + "\n" for value in values),
        encoding="utf-8",
    )


def _sample(rank: int, timestamp: int, node: str, power: float) -> NodeTelemetrySample:
    return NodeTelemetrySample(
        schema_version=1,
        monotonic_timestamp_ns=timestamp,
        utc_timestamp="2026-07-24T00:00:00+00:00",
        gpu_uuid_raw=f"GPU-{UUIDS[rank].upper()}",
        gpu_uuid=UUIDS[rank],
        node_identity=node,
        power_watts=power,
        gpu_utilization_percent=50.0,
        memory_utilization_percent=25.0,
        allocated_device_memory_bytes=1024,
        temperature_celsius=40.0,
        graphics_clock_mhz=1200,
        memory_clock_mhz=900,
        cumulative_device_energy_millijoules=None,
        sampling_backend="nvml",
        configured_interval_ms=100,
        sampling_error_status=None,
        execution_attempt=1,
        slurm_allocation_identity="323850_0:323851",
    )


def _append_interval(
    recorder: IntervalRecorder,
    *,
    rank: int,
    sequence: int,
    category: str,
    start_ns: int,
    end_ns: int,
    identity: dict | None = None,
) -> None:
    recorder._append(
        IntervalRecord(
            interval_id=f"attempt-1-rank-{rank}-interval-{sequence}",
            category=category,
            execution_attempt=1,
            gpu_uuid=UUIDS[rank],
            start_ns=start_ns,
            end_ns=end_ns,
            wall_seconds=(end_ns - start_ns) / 1_000_000_000,
            accepted=True,
            identity=identity,
            source_rank=rank,
            schema_version=2,
        )
    )


def _case(
    root: Path,
    *,
    second_node_offset_ns: int,
    one_node: bool = False,
    reverse_rank_rows: bool = False,
) -> tuple[ComparativeMeasurementSession, list[dict], dict]:
    run_dir = root / "run"
    attempt_dir = run_dir / "measurement_attempts" / "attempt_1"
    process_mappings = []
    idle_records = []
    telemetry_by_node: dict[str, list[NodeTelemetrySample]] = {}
    expected_client_energy = {}
    expected_phase_energy = 0.0
    expected_validation_energy = 0.0
    node_count = 1 if one_node else 2

    for rank in range(4):
        node_rank = 0 if one_node else rank // 2
        node = f"node-{node_rank}"
        origin = 1_000_000_000 + (second_node_offset_ns if node_rank == 1 else 0)
        power = 100.0 + 10.0 * rank
        process_mappings.append({"rank": rank, "node_rank": node_rank, "host": node, "gpu_uuid": UUIDS[rank]})
        rank_dir = attempt_dir / f"rank_{rank}"
        recorder = IntervalRecorder(
            rank_dir / "execution_intervals.jsonl",
            source_rank=rank,
            schema_version=2,
        )
        client_start = origin + 100_000_000 + rank * 5_000_000
        client_end = client_start + 100_000_000
        identity = ClientIntervalIdentity(
            "shd",
            "simulated_two_node_four_rank",
            37,
            1,
            rank,
            f"client_{rank}",
            1000 + rank,
            1,
            UUIDS[rank],
        )
        client_interval = recorder.record_client(
            identity,
            client_start,
            client_end,
            data_wait_seconds=0.0,
            cuda_event_seconds=0.05,
        )
        _append_interval(
            recorder,
            rank=rank,
            sequence=2,
            category="model_distribution",
            start_ns=origin + 225_000_000,
            end_ns=origin + 275_000_000,
            identity={"round_number": 1},
        )
        next_sequence = 3
        _append_interval(
            recorder,
            rank=rank,
            sequence=next_sequence,
            category="result_collection",
            start_ns=origin + 300_000_000,
            end_ns=origin + 350_000_000,
            identity={"round_number": 1},
        )
        next_sequence += 1
        if rank == 0:
            _append_interval(
                recorder,
                rank=rank,
                sequence=next_sequence,
                category="aggregation",
                start_ns=origin + 375_000_000,
                end_ns=origin + 425_000_000,
                identity={"round_number": 1},
            )
            next_sequence += 1
            _append_interval(
                recorder,
                rank=rank,
                sequence=next_sequence,
                category="validation",
                start_ns=origin + 450_000_000,
                end_ns=origin + 500_000_000,
                identity={"round_number": 1},
            )
            next_sequence += 1
            expected_validation_energy = power * 0.05
            for category, start in (
                ("checkpoint_writing", 525_000_000),
                ("official_test", 600_000_000),
            ):
                _append_interval(
                    recorder,
                    rank=rank,
                    sequence=next_sequence,
                    category=category,
                    start_ns=origin + start,
                    end_ns=origin + start + 50_000_000,
                    identity={"official_test": True} if category == "official_test" else {"round_number": 1},
                )
                next_sequence += 1
        _append_interval(
            recorder,
            rank=rank,
            sequence=next_sequence,
            category="complete_treatment",
            start_ns=origin + 50_000_000,
            end_ns=origin + 700_000_000,
        )
        resource = {
            **identity.__dict__,
            "schema_version": 2,
            "interval_id": client_interval.interval_id,
            "interval_source_rank": rank,
            "interval_source_node_rank": node_rank,
            "process_rank": rank,
            "node_rank": node_rank,
            "gross_energy_joules": None,
            "idle_adjusted_energy_joules": None,
            "accepted": False,
            "exclusion_reason": "energy_pending",
        }
        _write_jsonl(rank_dir / "client_resource_records.jsonl", [resource])
        if reverse_rank_rows:
            interval_path = rank_dir / "execution_intervals.jsonl"
            _write_jsonl(interval_path, list(reversed(_jsonl(interval_path))))
        expected_client_energy[rank] = power * 0.1
        expected_phase_energy += power * 0.05
        for timestamp in range(0, 751_000_000, 25_000_000):
            telemetry_by_node.setdefault(node, []).append(_sample(rank, origin + timestamp, node, power))

    node_rank_by_name = {value["host"]: value["node_rank"] for value in process_mappings}
    for node, values in telemetry_by_node.items():
        values.sort(key=lambda value: (value.monotonic_timestamp_ns, value.gpu_uuid))
        _write_jsonl(
            attempt_dir / f"node_{node_rank_by_name[node]}_samples.jsonl",
            [value.record() for value in values],
        )
        origin = min(value.monotonic_timestamp_ns for value in values)
        idle_records.append(
            {
                "node_rank": node_rank_by_name[node],
                "node_identity": node,
                "idle_before": {"start_ns": origin, "end_ns": origin + 50_000_000},
                "idle_after": {
                    "start_ns": origin + 700_000_000,
                    "end_ns": origin + 750_000_000,
                },
            }
        )
    scientific = [
        {
            "round_number": 1,
            "selected_position": rank,
            "client_id": f"client_{rank}",
            "resolved_training_seed": 1000 + rank,
            "process_rank": rank,
        }
        for rank in range(4)
    ]
    _write_jsonl(run_dir / "client_metrics.jsonl", scientific)

    session = ComparativeMeasurementSession.__new__(ComparativeMeasurementSession)
    session.run_dir = run_dir
    session.attempt_dir = attempt_dir
    session.execution_attempt = 1
    session.allocation_identity = "323850_0:323851"
    session.process_mappings = list(reversed(process_mappings)) if reverse_rank_rows else process_mappings
    session.context = SimpleNamespace(
        world_size=4,
        node_count=node_count,
        physical_device_count=4,
    )
    session.config = {
        "dataset": {"name": "shd"},
        "metadata": {"experiment": "simulated_two_node_four_rank"},
        "seed": 37,
        "federated": {"rounds": 1, "clients_per_round": 4, "checkpoint_selection": "best_validation"},
        "aggregation_execution": {"topology": "flat_ordered"},
        "comparative_evaluation": {"evidence_complete_outcome": "test_only_complete"},
    }
    session.bundle = SimpleNamespace(official_test_access_count=1)
    session.measurement = {
        "maximum_sample_gap_ms": 250,
        "boundary_reconciliation_tolerance_joules": 1e-9,
    }
    expected = {
        "client": expected_client_energy,
        "model_distribution": expected_phase_energy,
        "validation": expected_validation_energy,
    }
    return session, idle_records, expected


def _finalize(case: tuple[ComparativeMeasurementSession, list[dict], dict]) -> tuple[dict, list[dict]]:
    session, idle_records, _ = case
    summary = session._finalize_attempt(True, idle_records)
    accepted = _jsonl(session.attempt_dir / "accepted_client_resource_records.jsonl")
    return summary, accepted


def _remove_phase(session: ComparativeMeasurementSession, rank: int, category: str) -> None:
    path = session.attempt_dir / f"rank_{rank}" / "execution_intervals.jsonl"
    values = _jsonl(path)
    removed = [value for value in values if value["category"] == category]
    assert len(removed) == 1
    _write_jsonl(path, [value for value in values if value["category"] != category])


def _append_phase(
    session: ComparativeMeasurementSession,
    rank: int,
    category: str,
    *,
    sequence: int = 99,
    identity: dict | None = None,
) -> None:
    path = session.attempt_dir / f"rank_{rank}" / "execution_intervals.jsonl"
    values = _jsonl(path)
    source = next(value for value in values if value["category"] == "model_distribution")
    values.append(
        {
            **source,
            "interval_id": f"attempt-1-rank-{rank}-interval-{sequence}",
            "category": category,
            "identity": {"round_number": 1} if identity is None else identity,
        }
    )
    _write_jsonl(path, values)


def test_four_rank_two_node_resolution_and_phase_energy_are_clock_domain_local(tmp_path: Path) -> None:
    synchronized = _case(tmp_path / "synchronized", second_node_offset_ns=0)
    skewed = _case(tmp_path / "skewed", second_node_offset_ns=OFFSET_NS)
    reordered = _case(
        tmp_path / "reordered",
        second_node_offset_ns=OFFSET_NS,
        reverse_rank_rows=True,
    )

    synchronized_summary, synchronized_rows = _finalize(synchronized)
    skewed_summary, skewed_rows = _finalize(skewed)
    reordered_summary, reordered_rows = _finalize(reordered)
    expected = skewed[2]

    client_ids = [value["interval_id"] for value in skewed_rows]
    assert len(client_ids) == len(set(client_ids)) == 4
    unqualified = [re.sub(r"-rank-\d+", "", value) for value in client_ids]
    assert unqualified == ["attempt-1-client-1"] * 4
    for rank, value in enumerate(skewed_rows):
        assert value["interval_source_rank"] == value["process_rank"] == rank
        assert value["gpu_uuid"] == UUIDS[rank]
        assert value["gross_energy_joules"] == pytest.approx(expected["client"][rank])

    old_last_write_wins = {}
    for rank in range(4):
        for interval in _jsonl(skewed[0].attempt_dir / f"rank_{rank}" / "execution_intervals.jsonl"):
            old_last_write_wins[re.sub(r"-rank-\d+", "", interval["interval_id"])] = interval
    wrong = old_last_write_wins["attempt-1-client-1"]
    assert wrong["source_rank"] == 3
    merged = [
        _sample(0, 1_000_000_000 + timestamp, "node-0", 100.0).device_sample()
        for timestamp in range(0, 401_000_000, 25_000_000)
    ]
    with pytest.raises(ValueError, match="leading|trailing"):
        integrate_energy(
            merged,
            int(wrong["start_ns"]),
            int(wrong["end_ns"]),
            idle_baseline_watts=100.0,
        )

    for summary in (synchronized_summary, skewed_summary, reordered_summary):
        phase = summary["phase_energy"]["model_distribution"]
        assert phase["gross_energy_joules"] == pytest.approx(expected["model_distribution"])
        assert phase["participating_device_interval_count"] == 4
        assert phase["per_device"].keys() == set(UUIDS)
        for rank, gpu_uuid in enumerate(UUIDS):
            assert phase["per_device"][gpu_uuid]["gross_energy_joules"] == pytest.approx((100.0 + 10.0 * rank) * 0.05)
        validation = summary["phase_energy"]["validation"]
        assert validation["gross_energy_joules"] == pytest.approx(expected["validation"])
        assert validation["participating_device_interval_count"] == 1
        assert set(validation["per_device"]) == {UUIDS[0]}
        assert "coordinator rank only" in validation["participation_policy"]
        assert summary["complete_treatment_energy"]["gross_energy_joules"] == pytest.approx(299.0)
        assert summary["complete_treatment_energy"]["idle_adjusted_energy_joules"] == pytest.approx(0.0)
        assert summary["accounted_energy"]["gross_energy_joules"] == pytest.approx(112.0)
        assert summary["unattributed_energy"]["gross_energy_joules"] == pytest.approx(187.0)
        assert summary["unattributed_energy"]["idle_adjusted_energy_joules"] == pytest.approx(0.0)
        assert all(
            abs(value) <= summary["energy_reconciliation"]["tolerance_joules"]
            for value in summary["energy_reconciliation"]["errors_joules"].values()
        )
    assert [value["gross_energy_joules"] for value in synchronized_rows] == pytest.approx(
        [value["gross_energy_joules"] for value in skewed_rows]
    )
    assert [value["gross_energy_joules"] for value in skewed_rows] == pytest.approx(
        [value["gross_energy_joules"] for value in reordered_rows]
    )
    assert skewed_summary["phase_energy"] == reordered_summary["phase_energy"]
    acceptance = skewed[0]._aggregate_attempts(True, skewed_summary)
    assert acceptance["accepted"]
    energy_summary = json.loads((skewed[0].run_dir / "energy_summary.json").read_text(encoding="utf-8"))
    assert energy_summary["accepted_client_record_count"] == 4
    assert energy_summary["schema_version"] == 2
    assert all(
        abs(value) <= energy_summary["energy_reconciliation"]["tolerance_joules"]
        for value in energy_summary["energy_reconciliation"]["errors_joules"].values()
    )


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("missing_resource_source", "missing or wrong interval source rank"),
        ("wrong_resource_source", "missing or wrong interval source rank"),
        ("missing_interval_source", "missing or wrong source rank"),
        ("duplicate_qualified_interval", "globally duplicated|qualified interval identity"),
        ("cross_attempt_interval", "crosses execution attempts"),
        ("previous_attempt_reference", "cannot be resolved exactly"),
        ("gpu_mismatch", "GPU differs from process mapping"),
    ],
)
def test_invalid_interval_relationships_fail_closed(
    tmp_path: Path,
    defect: str,
    message: str,
) -> None:
    session, idle_records, _ = _case(tmp_path / defect, second_node_offset_ns=OFFSET_NS)
    resource_path = session.attempt_dir / "rank_0" / "client_resource_records.jsonl"
    interval_path = session.attempt_dir / "rank_0" / "execution_intervals.jsonl"
    resources = _jsonl(resource_path)
    intervals = _jsonl(interval_path)
    if defect == "missing_resource_source":
        resources[0].pop("interval_source_rank")
    elif defect == "wrong_resource_source":
        resources[0]["interval_source_rank"] = 1
    elif defect == "missing_interval_source":
        intervals[0].pop("source_rank")
    elif defect == "duplicate_qualified_interval":
        intervals.append(dict(intervals[0]))
    elif defect == "cross_attempt_interval":
        intervals[0]["execution_attempt"] = 2
    elif defect == "previous_attempt_reference":
        resources[0]["interval_id"] = "attempt-0-rank-0-client-1"
    elif defect == "gpu_mismatch":
        resources[0]["gpu_uuid"] = UUIDS[1]
    _write_jsonl(resource_path, resources)
    _write_jsonl(interval_path, intervals)

    with pytest.raises(ValueError, match=message):
        session._finalize_attempt(True, idle_records)
    assert not (session.attempt_dir / "attempt_energy_summary.json").exists()
    assert not (session.run_dir / "measurement_acceptance.json").exists()
    assert not (session.run_dir / "energy_summary.json").exists()
    assert not (session.run_dir / "allocation_completion.json").exists()


def test_one_node_multi_gpu_still_resolves_each_rank_when_clocks_are_close(tmp_path: Path) -> None:
    case = _case(tmp_path, second_node_offset_ns=0, one_node=True)
    summary, rows = _finalize(case)
    assert [value["interval_source_rank"] for value in rows] == [0, 1, 2, 3]
    assert [value["gpu_uuid"] for value in rows] == list(UUIDS)
    assert [value["gross_energy_joules"] for value in rows] == pytest.approx(
        [case[2]["client"][rank] for rank in range(4)]
    )
    assert summary["complete_treatment_energy"]["device_count"] == 4
    assert summary["node_count"] == 1
    old_last_write_wins = {}
    for rank in range(4):
        for interval in _jsonl(case[0].attempt_dir / f"rank_{rank}" / "execution_intervals.jsonl"):
            old_last_write_wins[re.sub(r"-rank-\d+", "", interval["interval_id"])] = interval
    wrong = old_last_write_wins["attempt-1-client-1"]
    assert wrong["source_rank"] == 3
    rank_zero_samples = [
        _sample(0, 1_000_000_000 + timestamp, "node-0", 100.0).device_sample()
        for timestamp in range(0, 401_000_000, 25_000_000)
    ]
    silently_covered = integrate_energy(
        rank_zero_samples,
        int(wrong["start_ns"]),
        int(wrong["end_ns"]),
        idle_baseline_watts=100.0,
    )
    assert silently_covered.gross_energy_joules == pytest.approx(10.0)

    wrong_session, wrong_idle, _ = _case(
        tmp_path / "wrong-rank",
        second_node_offset_ns=0,
        one_node=True,
    )
    path = wrong_session.attempt_dir / "rank_0" / "client_resource_records.jsonl"
    resources = _jsonl(path)
    resources[0]["interval_source_rank"] = 1
    _write_jsonl(path, resources)
    with pytest.raises(ValueError, match="missing or wrong interval source rank"):
        wrong_session._finalize_attempt(True, wrong_idle)


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("missing_model_worker", "model_distribution phase participation differs"),
        ("missing_collection_worker", "result_collection phase participation differs"),
        ("unexpected_worker_validation", "validation phase participation differs"),
        ("unexpected_worker_checkpoint", "checkpoint_writing phase participation differs"),
        ("duplicate_coordinator_aggregation", "aggregation phase duplicates participant rank 0"),
        ("missing_complete_treatment", "complete treatment interval is missing or duplicated for rank 3"),
        ("wrong_category_participant", "aggregation phase participation differs"),
        ("wrong_round_identity", "aggregation has an unexpected phase identity"),
        ("rejected_expected_phase", "model_distribution phase uses a rejected participant interval"),
        ("mixed_interval_schema", "interval does not use comparative schema version 2"),
        ("mixed_resource_schema", "client row does not use comparative schema version 2"),
    ],
)
def test_phase_participation_and_schema_defects_fail_before_publication(
    tmp_path: Path,
    defect: str,
    message: str,
) -> None:
    session, idle_records, _ = _case(tmp_path / defect, second_node_offset_ns=OFFSET_NS)
    if defect == "missing_model_worker":
        _remove_phase(session, 2, "model_distribution")
    elif defect == "missing_collection_worker":
        _remove_phase(session, 3, "result_collection")
    elif defect == "unexpected_worker_validation":
        _append_phase(session, 1, "validation")
    elif defect == "unexpected_worker_checkpoint":
        _append_phase(session, 1, "checkpoint_writing")
    elif defect == "duplicate_coordinator_aggregation":
        _append_phase(session, 0, "aggregation")
    elif defect == "missing_complete_treatment":
        _remove_phase(session, 3, "complete_treatment")
    elif defect == "wrong_category_participant":
        _append_phase(session, 2, "aggregation")
    elif defect == "wrong_round_identity":
        path = session.attempt_dir / "rank_0" / "execution_intervals.jsonl"
        values = _jsonl(path)
        next(value for value in values if value["category"] == "aggregation")["identity"] = {"round_number": 2}
        _write_jsonl(path, values)
    elif defect == "rejected_expected_phase":
        path = session.attempt_dir / "rank_1" / "execution_intervals.jsonl"
        values = _jsonl(path)
        next(value for value in values if value["category"] == "model_distribution")["accepted"] = False
        _write_jsonl(path, values)
    elif defect == "mixed_interval_schema":
        path = session.attempt_dir / "rank_1" / "execution_intervals.jsonl"
        values = _jsonl(path)
        values[0]["schema_version"] = 1
        _write_jsonl(path, values)
    elif defect == "mixed_resource_schema":
        path = session.attempt_dir / "rank_1" / "client_resource_records.jsonl"
        values = _jsonl(path)
        values[0]["schema_version"] = 1
        _write_jsonl(path, values)

    with pytest.raises((RuntimeError, ValueError), match=message):
        session._finalize_attempt(True, idle_records)
    for name in (
        "accepted_client_resource_records.jsonl",
        "attempt_energy_summary.json",
        "attempt_evidence_complete.json",
    ):
        assert not (session.attempt_dir / name).exists()
    for name in (
        "client_resource_records.jsonl",
        "energy_summary.json",
        "measurement_acceptance.json",
        "allocation_completion.json",
    ):
        assert not (session.run_dir / name).exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("interval_source_node_rank", 1, "wrong interval source node rank"),
        ("node_rank", 1, "wrong node rank"),
        ("process_rank", 1, "wrong process rank"),
    ],
)
def test_client_rank_node_process_provenance_is_exact(
    tmp_path: Path,
    field: str,
    value: int,
    message: str,
) -> None:
    session, idle_records, _ = _case(tmp_path, second_node_offset_ns=OFFSET_NS)
    path = session.attempt_dir / "rank_0" / "client_resource_records.jsonl"
    rows = _jsonl(path)
    rows[0][field] = value
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match=message):
        session._finalize_attempt(True, idle_records)


def test_hierarchical_policy_matches_uninstrumented_collection_and_aggregation(tmp_path: Path) -> None:
    session, idle_records, _ = _case(tmp_path, second_node_offset_ns=OFFSET_NS)
    session.config["aggregation_execution"]["topology"] = "node_hierarchical"
    for rank in range(4):
        _remove_phase(session, rank, "result_collection")
    _remove_phase(session, 0, "aggregation")
    summary = session._finalize_attempt(True, idle_records)
    assert summary["phase_energy"]["result_collection"]["logical_phase_count"] == 0
    assert summary["phase_energy"]["aggregation"]["logical_phase_count"] == 0
    assert "no interval" in summary["phase_energy"]["aggregation"]["participation_policy"]


def test_aggregate_reconciliation_rejects_omitted_or_duplicated_phase_energy(tmp_path: Path) -> None:
    session, idle_records, _ = _case(tmp_path, second_node_offset_ns=OFFSET_NS)
    summary = session._finalize_attempt(True, idle_records)
    path = session.attempt_dir / "attempt_energy_summary.json"
    persisted = json.loads(path.read_text(encoding="utf-8"))
    persisted["phase_energy"]["aggregation"]["gross_energy_joules"] += 1.0
    path.write_text(json.dumps(persisted), encoding="utf-8")
    with pytest.raises(ValueError, match="accounted energy omits or duplicates"):
        session._aggregate_attempts(True, summary)
    assert not (session.run_dir / "client_resource_records.jsonl").exists()
    assert not (session.run_dir / "measurement_acceptance.json").exists()


def test_official_test_phase_and_access_are_coordinator_owned_once(tmp_path: Path) -> None:
    session, idle_records, _ = _case(tmp_path, second_node_offset_ns=OFFSET_NS)
    summary = session._finalize_attempt(True, idle_records)
    phase = summary["phase_energy"]["official_test"]
    assert phase["validated_participation"] == [
        {"identity": {"official_test": True}, "process_ranks": [0]}
    ]
    (session.run_dir / "official_test_metrics.json").write_text(
        json.dumps({"access_count": 1, "evaluation_completed": True}), encoding="utf-8"
    )
    acceptance = session._aggregate_attempts(True, summary)
    assert acceptance["accepted"]
    energy = json.loads((session.run_dir / "energy_summary.json").read_text(encoding="utf-8"))
    assert energy["official_test_access_count"] == 1


def test_partial_attempt_files_without_completion_marker_are_rejected(tmp_path: Path) -> None:
    session, idle_records, _ = _case(tmp_path, second_node_offset_ns=OFFSET_NS)
    summary = session._finalize_attempt(True, idle_records)
    (session.attempt_dir / "attempt_evidence_complete.json").unlink()
    with pytest.raises(ValueError, match="partial comparative attempt evidence"):
        session._aggregate_attempts(True, summary)
    assert not (session.run_dir / "measurement_acceptance.json").exists()


def test_reconciliation_tolerance_cannot_be_excessively_permissive(tmp_path: Path) -> None:
    session, idle_records, _ = _case(tmp_path, second_node_offset_ns=OFFSET_NS)
    session.measurement["boundary_reconciliation_tolerance_joules"] = 1.000001e-6
    with pytest.raises(ValueError, match="excessively permissive"):
        session._finalize_attempt(True, idle_records)
    assert not (session.attempt_dir / "accepted_client_resource_records.jsonl").exists()
