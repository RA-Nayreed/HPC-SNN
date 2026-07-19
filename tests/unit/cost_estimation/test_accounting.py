import pytest

from fedapfa.cost_estimation.artifacts import _validate_slurm_accounting


def test_slurm_accounting_reports_allocated_time_without_energy_inference(tmp_path):
    path = tmp_path / "accounting.txt"
    path.write_text(
        "JobID|State|ExitCode|ElapsedRaw|AllocTRES|Start|End\n"
        "123|COMPLETED|0:0|7200|cpu=72,gres/gpu=1,gres/gpu:gh200=1|2026-01-01T00:00:00|2026-01-01T02:00:00\n"
    )
    value = _validate_slurm_accounting(path, "123")
    assert value["elapsed_seconds"] == 7200
    assert value["allocated_gpu_hours"] == 2.0
    assert value["pending_time_seconds"] is None
    assert not value["device_energy_inferred_from_allocation"]


def test_slurm_accounting_sums_resumed_allocations(tmp_path):
    path = tmp_path / "accounting.txt"
    path.write_text(
        "JobID|State|ExitCode|ElapsedRaw|AllocTRES|Start|End\n"
        "123|TIMEOUT|0:0|3600|cpu=72,gres/gpu=1|a|b\n"
        "124|COMPLETED|0:0|7200|cpu=72,gres/gpu=1|c|d\n"
    )
    value = _validate_slurm_accounting(path, {"123", "124"})
    assert value["elapsed_seconds"] == 10800
    assert value["allocated_gpu_hours"] == 3.0
    assert value["allocation_ids"] == ["123", "124"]


def test_slurm_accounting_rejects_unsuccessful_allocation(tmp_path):
    path = tmp_path / "accounting.txt"
    path.write_text(
        "JobID|State|ExitCode|ElapsedRaw|AllocTRES|Start|End\n"
        "123|FAILED|1:0|10|gres/gpu=1|a|b\n"
    )
    with pytest.raises(ValueError, match="did not complete"):
        _validate_slurm_accounting(path, "123")
