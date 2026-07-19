import json

import pytest

from fedapfa.measurement.client_interval import ClientIntervalIdentity, IntervalRecord, IntervalRecorder
from fedapfa.measurement.clock import CpuTimingAdapter
from fedapfa.measurement.runtime import _next_execution_attempt


class Clock:
    def __init__(self, values):
        self.values = iter(values)

    def now_ns(self):
        return next(self.values)


def _identity(attempt=1, position=0):
    return ClientIntervalIdentity("shd", "resource", 7, 1, position, "client_00", 91, attempt, "GPU-a")


def test_client_identity_uniqueness_nonoverlap_and_resumed_attempt(tmp_path):
    recorder = IntervalRecorder(tmp_path / "intervals.jsonl")
    first = recorder.record_client(_identity(), 0, 10_000_000, 0.002, 0.003)
    assert first.residual_host_seconds == pytest.approx(0.005)
    with pytest.raises(ValueError, match="duplicated"):
        recorder.record_client(_identity(), 10_000_000, 20_000_000, 0.002, 0.003)
    with pytest.raises(ValueError, match="overlap"):
        recorder.record_client(_identity(position=1), 9_000_000, 20_000_000, 0.002, 0.003)
    resumed = recorder.record_client(_identity(attempt=2, position=1), 20_000_000, 30_000_000, 0.002, 0.003)
    assert resumed.accepted and resumed.execution_attempt == 2


def test_timing_reconciliation_and_incomplete_interval_exclusion(tmp_path):
    with pytest.raises(ValueError, match="reconcile"):
        IntervalRecord("x", "client_training", 1, "GPU-a", 0, 10_000_000, 0.01, True, None, {}, 0.01, 0.01, 0.0)
    recorder = IntervalRecorder(tmp_path / "intervals.jsonl", Clock([0, 4_000_000]))
    with pytest.raises(RuntimeError):
        with recorder.interval("communication_round", 1, "GPU-a"):
            raise RuntimeError("interrupt")
    value = json.loads((tmp_path / "intervals.jsonl").read_text().strip())
    assert not value["accepted"]
    assert value["exclusion_reason"] == "RuntimeError"


def test_cpu_timing_adapter_uses_injected_monotonic_clock():
    adapter = CpuTimingAdapter(Clock([0, 2_000_000, 7_000_000, 10_000_000]))
    adapter.start()
    token = adapter.begin_device_work()
    adapter.end_device_work(token)
    result = adapter.finish()
    assert result.start_ns == 0 and result.end_ns == 10_000_000
    assert result.cuda_seconds == pytest.approx(0.005)


def test_execution_attempt_advances_from_partial_sample_record(tmp_path):
    (tmp_path / "device_samples.jsonl").write_text(
        json.dumps({"execution_attempt": 4}) + "\n",
        encoding="utf-8",
    )
    assert _next_execution_attempt(tmp_path, []) == 5


def test_execution_attempt_advances_from_sampler_startup_attempt_record(tmp_path):
    (tmp_path / "calibration_reference.json").write_text(
        json.dumps({"schema_version": 1, "attempts": [{"execution_attempt": 3}]}),
        encoding="utf-8",
    )
    assert _next_execution_attempt(tmp_path, []) == 4
