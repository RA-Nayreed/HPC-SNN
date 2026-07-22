from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fedapfa.analysis.comparative_summary import (
    classify_numerical_evidence,
    load_allocation_timing,
    paired_scaling_metrics,
    parse_slurm_accounting,
    reconcile_allocation,
    sample_statistics,
)


def test_sample_standard_deviation_and_scaling_formulas() -> None:
    assert sample_statistics([1, 2, 3]) == {"mean": 2.0, "sample_standard_deviation": 1.0}
    assert paired_scaling_metrics(10, 4, 2) == {"speedup": 2.5, "parallel_efficiency": 1.25}


def test_numerical_classification_is_not_accuracy_similarity() -> None:
    reference = {"weight": torch.tensor([1.0, 2.0])}
    candidate = {"weight": torch.tensor([1.0, 2.000001])}
    result = classify_numerical_evidence(
        reference_state=reference,
        candidate_state=candidate,
        reference_selected_clients=[["a", "b"]],
        candidate_selected_clients=[["a", "b"]],
        reference_client_seeds=[1, 2],
        candidate_client_seeds=[1, 2],
        reference_ordered_updates=[["u1", "u2"]],
        candidate_ordered_updates=[["u1", "u2"]],
        reference_weights=[[0.5, 0.5]],
        candidate_weights=[[0.5, 0.5]],
        reference_checkpoint_round=3,
        candidate_checkpoint_round=3,
        reference_predictions=[0, 1],
        candidate_predictions=[0, 1],
        reference_metrics={"accuracy": 0.5, "macro_f1": 0.5},
        candidate_metrics={"accuracy": 0.5, "macro_f1": 0.5},
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
    )
    assert result["structural_identity"]
    assert result["bounded_parameter_identity"]
    assert not result["exact_parameter_identity"]
    assert not result["checkpoint_identity"]
    assert not result["exact_equivalence"]


def test_allocation_accounting_keeps_display_and_raw_ids(tmp_path: Path) -> None:
    path = tmp_path / "accounting.txt"
    path.write_text(
        "JobID|JobIDRaw|State|ExitCode|ElapsedRaw|AllocTRES|Start|End|NodeList\n"
        "900_0|901|COMPLETED|0:0|100|cpu=288,gres/gpu:gh200=4|2026-01-01T00:00:00+00:00|2026-01-01T00:01:40+00:00|r01\n",
        encoding="utf-8",
    )
    records, digest = parse_slurm_accounting(path)
    assert len(digest) == 64
    assert records["900_0"]["raw_slurm_job_id"] == "901"
    assert records["900_0"]["billed_gpu_hours"] == pytest.approx(100 * 4 / 3600)


def test_sequential_allocation_reconciliation_does_not_duplicate_billing() -> None:
    accounting = {
        "display_array_task_id": "900_0",
        "raw_slurm_job_id": "901",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed_seconds": 100,
        "allocated_tres": "gres/gpu=4",
        "start": "start",
        "end": "end",
        "start_unix_seconds": 0,
        "end_unix_seconds": 100,
        "node_list": "r01",
        "physical_gpu_count": 4,
        "billed_gpu_hours": 100 * 4 / 3600,
    }
    order = ("iid", "dirichlet_alpha_1_0", "dirichlet_alpha_0_5", "dirichlet_alpha_0_1")
    runs = [
        {
            "dataset": "shd",
            "seed": 37,
            "treatment_id": treatment,
            "treatment_position": index + 1,
            "runtime_seconds": 20,
            "derived_gpu_exposure_hours": 20 * 4 / 3600,
            "execution_start_unix_seconds": 5 + index * 22,
            "execution_end_unix_seconds": 25 + index * 22,
        }
        for index, treatment in enumerate(order)
    ]
    result = reconcile_allocation(accounting, runs, expected_order=order)
    assert result["allocation_initialization_seconds"] == 5
    assert result["between_treatment_seconds"] == 6
    assert result["remaining_allocation_overhead_seconds"] == 9
    assert result["allocation_reconciliation_error_seconds"] == 0
    assert "billed_gpu_hours" not in result["treatments"][0]


def test_launcher_allocation_timing_requires_complete_monotonic_order(tmp_path: Path) -> None:
    timing = {
        "schema_version": 1,
        "allocation_index": 0,
        "display_array_task_id": "900_0",
        "raw_slurm_job_id": "901",
        "execution_order": ["iid", "dirichlet_alpha_1_0"],
        "launcher_observed_allocation_start": {"monotonic_timestamp_ns": 1},
        "launcher_observed_allocation_end": {"monotonic_timestamp_ns": 10},
        "launcher_observed_allocation_duration_seconds": 9e-9,
        "open_treatment": None,
        "completed": True,
        "treatments": [
            {
                "treatment_id": "iid",
                "treatment_position": 1,
                "start": {"monotonic_timestamp_ns": 2},
                "end": {"monotonic_timestamp_ns": 4},
                "launcher_observed_duration_seconds": 2e-9,
            },
            {
                "treatment_id": "dirichlet_alpha_1_0",
                "treatment_position": 2,
                "start": {"monotonic_timestamp_ns": 5},
                "end": {"monotonic_timestamp_ns": 9},
                "launcher_observed_duration_seconds": 4e-9,
            },
        ],
    }
    path = tmp_path / "timing.json"
    path.write_text(__import__("json").dumps(timing), encoding="utf-8")
    assert load_allocation_timing(
        path,
        allocation_index=0,
        display_array_task_id="900_0",
        expected_order=("iid", "dirichlet_alpha_1_0"),
    )["completed"]
    timing["treatments"][1]["start"]["monotonic_timestamp_ns"] = 3
    path.write_text(__import__("json").dumps(timing), encoding="utf-8")
    with pytest.raises(ValueError, match="nonmonotonic"):
        load_allocation_timing(
            path,
            allocation_index=0,
            display_array_task_id="900_0",
            expected_order=("iid", "dirichlet_alpha_1_0"),
        )


def test_negative_or_nonmonotonic_result_does_not_enter_execution_acceptance() -> None:
    metrics = paired_scaling_metrics(10, 12, 4)
    assert metrics["speedup"] < 1
    assert metrics["parallel_efficiency"] < 1


def test_resumable_interruption_accounting_is_billed_once(tmp_path: Path) -> None:
    path = tmp_path / "interrupted-accounting.txt"
    path.write_text(
        "JobID|JobIDRaw|State|ExitCode|ElapsedRaw|AllocTRES|Start|End|NodeList\n"
        "910_0|911|TIMEOUT|0:1|120|cpu=288,gres/gpu:gh200=4|"
        "2026-01-01T00:00:00+00:00|2026-01-01T00:02:00+00:00|r01\n",
        encoding="utf-8",
    )
    records, _ = parse_slurm_accounting(path)
    record = records["910_0"]
    assert record["resumable_interruption"] is True
    assert record["state_base"] == "TIMEOUT"
    assert record["billed_gpu_hours"] == pytest.approx(120 * 4 / 3600)


def test_interrupted_launcher_timing_reconciles_unclassified_tail(tmp_path: Path) -> None:
    order = ("iid", "dirichlet_alpha_1_0", "dirichlet_alpha_0_5", "dirichlet_alpha_0_1")
    timing = {
        "schema_version": 1,
        "allocation_index": 0,
        "display_array_task_id": "910_0",
        "raw_slurm_job_id": "911",
        "execution_order": list(order),
        "launcher_observed_allocation_start": {"monotonic_timestamp_ns": 1},
        "launcher_observed_allocation_end": None,
        "treatments": [],
        "open_treatment": {
            "treatment_id": "iid",
            "treatment_position": 1,
            "start": {"monotonic_timestamp_ns": 2},
            "end": None,
            "launcher_observed_duration_seconds": None,
        },
        "completed": False,
    }
    path = tmp_path / "interrupted-timing.json"
    path.write_text(__import__("json").dumps(timing), encoding="utf-8")
    validated = load_allocation_timing(
        path,
        allocation_index=0,
        display_array_task_id="910_0",
        expected_order=order,
    )
    accounting = {
        "display_array_task_id": "910_0",
        "raw_slurm_job_id": "911",
        "state": "TIMEOUT",
        "state_base": "TIMEOUT",
        "exit_code": "0:1",
        "resumable_interruption": True,
        "elapsed_seconds": 100,
        "allocated_tres": "gres/gpu=4",
        "start": "start",
        "end": "end",
        "start_unix_seconds": 0,
        "end_unix_seconds": 100,
        "node_list": "r01",
        "physical_gpu_count": 4,
        "billed_gpu_hours": 100 * 4 / 3600,
    }
    runs = [
        {
            "dataset": "shd",
            "seed": 37,
            "treatment_id": treatment,
            "treatment_position": position,
            "runtime_seconds": 20,
            "derived_gpu_exposure_hours": 20 * 4 / 3600,
            "execution_start_unix_seconds": 0,
            "execution_end_unix_seconds": 20,
        }
        for position, treatment in enumerate(order, start=1)
    ]
    result = reconcile_allocation(accounting, runs, expected_order=order, launcher_timing=validated)
    assert result["launcher_timing_completed"] is False
    assert result["treatments"][0]["timing_status"] == "interrupted_not_closed"
    assert result["interrupted_unclassified_seconds"] == pytest.approx(99.999999999)
    assert result["allocation_reconciliation_error_seconds"] == pytest.approx(0.0)
