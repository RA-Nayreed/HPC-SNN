import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import fedapfa.measurement.runtime as runtime
from fedapfa.measurement.power import NvmlProcessSampler, PowerSampler


class Adapter:
    uuid = "GPU-injected"

    def sample(self, _timestamp, _utc, _interval):
        raise NotImplementedError

    def close(self):
        return None


def _session(tmp_path, monkeypatch, adapter=None):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps({"passed": True, "sampling_interval_ms": 100}),
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "_cuda_uuid", lambda _device: "GPU-production")
    monkeypatch.setattr(runtime, "git_metadata", lambda: {"commit": "test"})
    context = SimpleNamespace(
        is_coordinator=True,
        world_size=1,
        visible_device_count=1,
        device=torch.device("cuda", 0),
    )
    config = {
        "resource_measurement": {"sampling_interval_ms": 100},
        "model": {"hidden_dims": [1, 1]},
    }
    return runtime.ResourceMeasurementSession(
        config,
        tmp_path,
        object(),
        nn.Linear(1, 1),
        context,
        calibration,
        adapter=adapter,
    )


def test_scientific_runtime_uses_spawned_nvml_sampler(tmp_path, monkeypatch):
    session = _session(tmp_path, monkeypatch)
    assert isinstance(session.sampler, NvmlProcessSampler)
    assert session.sampler._context.get_start_method() == "spawn"
    assert session.sampler.interval_ms == 100
    assert session.sampler.adapter_factory.expected_uuid == "GPU-production"


def test_runtime_keeps_injectable_adapter_path(tmp_path, monkeypatch):
    adapter = Adapter()
    session = _session(tmp_path, monkeypatch, adapter)
    assert isinstance(session.sampler, PowerSampler)
    assert session.sampler.adapter is adapter


def test_runtime_rejects_duplicate_interval_identity_before_lookup(tmp_path):
    session = runtime.ResourceMeasurementSession.__new__(runtime.ResourceMeasurementSession)
    session.run_dir = tmp_path
    interval = {"interval_id": "attempt-1-client-1"}
    (tmp_path / "execution_intervals.jsonl").write_text(
        json.dumps(interval) + "\n" + json.dumps(interval) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing or duplicated"):
        session._finalize(True, None)
