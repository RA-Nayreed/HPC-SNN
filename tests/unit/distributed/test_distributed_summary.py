import copy
import json

import pytest
import torch
import yaml

from fedapfa.cli.summarize_distributed_evaluation import (
    _comparison,
    _parameter_difference,
    summarize_distributed_evaluation,
)
from fedapfa.configuration import (
    distributed_execution_identity,
    distributed_scientific_identity,
    load_device_capacity_manifest,
    load_distributed_evaluation_manifest,
)

MANIFEST = "experiments/distributed_evaluation/manifest.yaml"
CAPACITY_MANIFEST = "experiments/device_capacity_evaluation/manifest.yaml"


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _create_run(root, task, task_index):
    parallel = task.config["parallel_execution"]
    devices = parallel["device_count"]
    processes = parallel["process_count"]
    selected_count = task.config["federated"]["clients_per_round"]
    run = root / f"{task.experiment}-seed{task.seed}"
    (run / "checkpoints").mkdir(parents=True)
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(task.config), encoding="utf-8")
    identity_suffix = f"{task.dataset}-{task.seed}"
    checkpoint_selection = task.config["federated"]["checkpoint_selection"]
    selected_artifact = "checkpoints/best.pt" if checkpoint_selection == "best_validation" else "checkpoints/last.pt"
    torch.save(
        {"global_model_state": {"weight": torch.tensor([float(task.seed), 1.0])}},
        run / selected_artifact,
    )
    _write_json(
        run / "acceptance.json",
        {
            "accepted": True,
            "completed": True,
            "scientific_status": (
                "equivalence_not_established" if task.dataset == "cifar10" else "not_claimed"
            ),
            "parallel_execution": parallel,
            "git_commit": "commit",
        },
    )
    selected_round = 100 if checkpoint_selection == "final_round" else 70
    test_metrics = {
        "accuracy": 0.8 + task.seed / 10000,
        "macro_f1": 0.79 + task.seed / 10000,
        "examples": 100,
    }
    _write_json(
        run / "final_metrics.json",
        {
            "completed": True,
            "parallel_execution": {**parallel, "process_mapping": []},
            "test": test_metrics,
            "best_validation_accuracy": (
                None if checkpoint_selection == "final_round" else 0.81 + task.seed / 10000
            ),
            "selected_round": selected_round,
            "checkpoint_selection": checkpoint_selection,
            "selected_checkpoint_artifact": selected_artifact,
            "logical_communication": {"cumulative_total_bytes": 123456},
            "execution_data_movement": {"total_bytes": devices * 1000},
            "split_id": f"split-{identity_suffix}",
            "partition_id": f"partition-{identity_suffix}",
            "model_initialization_id": f"initialization-{identity_suffix}",
            "scientific_identity": distributed_scientific_identity(task.config),
            "execution_identity": distributed_execution_identity(task.config),
            "aggregation_weighting": task.config["federated"]["aggregation_weighting"],
            "data_protocol": {"official_test_access_count": 1},
        },
    )
    _write_json(
        run / "official_test_metrics.json",
        {
            "access_count": 1,
            "evaluation_completed": True,
            "evaluated_after_model_selection": True,
            "monitored_during_rounds": False,
            "complete_split": True,
            "dataset_identity": {"name": f"official-{identity_suffix}"},
            "metrics": test_metrics,
        },
    )
    seconds = {1: 1.0, 2: 0.6, 4: 0.4}[devices]
    selected = [f"client_{index:02d}" for index in range(selected_count)]
    rounds = []
    measurements = []
    client_records = []
    for round_number in range(1, 101):
        rounds.append(
            {
                "round_number": round_number,
                "selected_client_ids": selected,
                "total_round_time_seconds": seconds,
                "parallel_client_training_wall_time_seconds": seconds / 2,
            }
        )
        for client_position in range(selected_count):
            client_records.append(
                {
                    "resolved_training_seed": round_number * 100 + client_position,
                    "example_count": 100 + client_position,
                }
            )
        measurements.append(
            {
                "round_number": round_number,
                "model_distribution_time_seconds": 0.01,
                "result_collection_time_seconds": 0.01,
                "aggregation_time_seconds": 0.01,
                "validation_time_seconds": 0.0 if checkpoint_selection == "final_round" else 0.01,
                "checkpoint_time_seconds": 0.01,
                "process_busy_time_seconds": {str(rank): seconds / 2 for rank in range(processes)},
                "process_load_imbalance": 0.0,
                "peak_cuda_memory_bytes_by_process_rank": {
                    str(rank): 1000 + rank for rank in range(processes)
                },
                "peak_cuda_reserved_bytes_by_process_rank": {
                    str(rank): 2000 + rank for rank in range(processes)
                },
            }
        )
    (run / "round_metrics.jsonl").write_text(
        "".join(json.dumps(value) + "\n" for value in rounds), encoding="utf-8"
    )
    (run / "client_metrics.jsonl").write_text(
        "".join(json.dumps(value) + "\n" for value in client_records), encoding="utf-8"
    )
    _write_json(
        run / "execution_measurements.json",
        {
            "completed": True,
            "rounds": measurements,
            "gpu_utilization": {
                "source": "nvidia-smi job-level physical-device samples",
                "sample_count": devices * 2,
                "mean_percent": 50.0,
                "minimum_percent": 40.0,
                "maximum_percent": 60.0,
                "by_device_index": {
                    str(device): {
                        "sample_count": 2,
                        "mean_percent": 50.0,
                        "minimum_percent": 40.0,
                        "maximum_percent": 60.0,
                    }
                    for device in range(devices)
                },
                "sampling_interval_seconds": 2,
                "execution_attempt": 1,
                "scope": "latest_execution_attempt",
            },
            "resource_allocation": {"array_job_id": "123", "array_task_id": str(task_index)},
        },
    )


def test_distributed_summary_uses_paired_reference_per_workload(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    for task_index, task in enumerate(load_distributed_evaluation_manifest(MANIFEST)):
        _create_run(runs, task, task_index)
    output = tmp_path / "results"
    summary = summarize_distributed_evaluation(MANIFEST, runs, output)
    assert summary["valid"]
    assert summary["completed_task_count"] == 24
    assert summary["reference_execution"] == "one_gpu_distributed_path_within_each_workload"
    assert not summary["cross_workload_pooling"]
    by_workload = {value["workload"]: value for value in summary["workloads"]}
    assert [len(by_workload[name]["treatments"]) for name in ("shd", "ssc", "cifar10")] == [3, 3, 2]
    shd = {value["device_count"]: value for value in by_workload["shd"]["treatments"]}
    assert shd[1]["paired_to_one_gpu"]["speedup"]["mean"] == 1.0
    assert shd[2]["paired_to_one_gpu"]["speedup"]["mean"] == pytest.approx(100 / 60)
    assert shd[4]["paired_to_one_gpu"]["speedup"]["mean"] == pytest.approx(2.5)
    assert shd[4]["numerical_equivalence_status"] == "exact"
    assert shd[4]["selected_round_status"] == "exact"
    assert shd[4]["metrics"]["gpu_utilization_percent"]["mean"] == 50.0
    assert {path.name for path in output.iterdir()} == {
        "distributed_evaluation_summary.json",
        "distributed_evaluation_summary.csv",
        "distributed_evaluation_summary.md",
    }


def test_distributed_summary_rejects_missing_mandatory_runs(tmp_path):
    output = tmp_path / "results"
    summary = summarize_distributed_evaluation(MANIFEST, tmp_path / "absent_runs", output)
    assert not summary["valid"]
    assert summary["completed_task_count"] == 0
    assert summary["validation_findings"]


def test_device_capacity_summary_reports_client_processes_per_device(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    tasks = load_device_capacity_manifest(CAPACITY_MANIFEST)
    for task_index, task in enumerate(tasks):
        _create_run(runs, task, task_index)
    summary = summarize_distributed_evaluation(CAPACITY_MANIFEST, runs, tmp_path / "results")
    assert summary["valid"] and summary["expected_task_count"] == 9
    treatments = summary["workloads"][0]["treatments"]
    assert [value["client_processes_per_device"] for value in treatments] == [1, 2, 4]
    assert all(value["device_count"] == 1 for value in treatments)


def test_distributed_summary_checks_slurm_device_allocations(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    tasks = load_distributed_evaluation_manifest(MANIFEST)
    for task_index, task in enumerate(tasks):
        _create_run(runs, task, task_index)
    accounting = tmp_path / "slurm-accounting.txt"
    accounting.write_text(
        "JobIDRaw|State|ExitCode|ElapsedRaw|AllocTRES\n"
        + "".join(
            f"123_{task_index}|COMPLETED|0:0|3600|cpu=288,gres/gpu:gh200="
            f"{task.config['parallel_execution']['device_count']}\n"
            for task_index, task in enumerate(tasks)
        ),
        encoding="utf-8",
    )
    summary = summarize_distributed_evaluation(MANIFEST, runs, tmp_path / "results", accounting)
    assert summary["valid"]
    assert summary["slurm_accounting_provided"]
    for workload in summary["workloads"]:
        for treatment in workload["treatments"]:
            assert treatment["metrics"]["allocated_gpu_hours"]["mean"] == treatment["device_count"]


def test_distributed_summary_accounts_for_resumed_allocations(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    tasks = load_distributed_evaluation_manifest(MANIFEST)
    for task_index, task in enumerate(tasks):
        _create_run(runs, task, task_index)

    resumed_task = tasks[0]
    resumed_path = runs / f"{resumed_task.experiment}-seed{resumed_task.seed}"
    measurements_path = resumed_path / "execution_measurements.json"
    measurements = json.loads(measurements_path.read_text(encoding="utf-8"))
    completing_allocation = measurements["resource_allocation"]
    measurements.update(
        {
            "resume_count": 1,
            "resource_allocations": [
                {"array_job_id": "122", "array_task_id": "0"},
                completing_allocation,
            ],
        }
    )
    measurements["gpu_utilization"]["execution_attempt"] = 2
    _write_json(measurements_path, measurements)

    accounting = tmp_path / "slurm-accounting.txt"
    accounting.write_text(
        "JobIDRaw|State|ExitCode|ElapsedRaw|AllocTRES\n"
        "122_0|TIMEOUT|0:0|1800|cpu=288,gres/gpu:gh200=1\n"
        + "".join(
            f"123_{task_index}|COMPLETED|0:0|3600|cpu=288,gres/gpu:gh200="
            f"{task.config['parallel_execution']['device_count']}\n"
            for task_index, task in enumerate(tasks)
        ),
        encoding="utf-8",
    )
    summary = summarize_distributed_evaluation(MANIFEST, runs, tmp_path / "results", accounting)
    assert summary["valid"]
    treatment = summary["workloads"][0]["treatments"][0]
    run = next(value for value in treatment["runs"] if value["seed"] == resumed_task.seed)
    assert run["allocation_states"] == ["TIMEOUT", "COMPLETED"]
    assert run["noncompleted_allocation_count"] == 1
    assert run["allocated_gpu_hours"] == 1.5
    assert "gpu_utilization_percent" not in treatment["metrics"]


def test_parameter_difference_is_finite_for_nonfloating_state():
    absolute, relative = _parameter_difference(
        {"counter": torch.tensor([0, 2], dtype=torch.int64)},
        {"counter": torch.tensor([1, 4], dtype=torch.int64)},
    )
    assert absolute == 2.0
    assert relative == 1.0 / torch.finfo(torch.float64).eps


def test_numerical_equivalence_requires_round_metrics_and_predictions():
    reference = {
        "scientific_identity": {"workload": "shd"},
        "split_id": "split",
        "partition_id": "partition",
        "model_initialization_id": "initialization",
        "selected_client_order": [["client_00"]],
        "client_training_seeds": [101],
        "client_example_counts": [8],
        "aggregation_weighting": "example_count",
        "logical_communication_bytes": 100,
        "checkpoint_selection": "best_validation",
        "official_test_access_count": 1,
        "official_test_examples": 2,
        "official_test_dataset_identity": {"sha256": "identity"},
        "git_commit": "commit",
        "selected_round": 7,
        "official_test_accuracy": 0.5,
        "official_test_macro_f1": 0.5,
        "_predictions": [0, 1],
        "_selected_state": {"weight": torch.tensor([1.0])},
    }
    exact = _comparison(reference, copy.deepcopy(reference))
    assert exact["numerical_equivalence_status"] == "exact"
    assert exact["prediction_agreement"] == 1.0

    another_round = copy.deepcopy(reference)
    another_round["selected_round"] = 8
    assert _comparison(reference, another_round)["numerical_equivalence_status"] == "difference_observed"

    another_prediction = copy.deepcopy(reference)
    another_prediction["_predictions"] = [1, 1]
    prediction_comparison = _comparison(reference, another_prediction)
    assert prediction_comparison["prediction_agreement"] == 0.5
    assert prediction_comparison["numerical_equivalence_status"] == "difference_observed"

    another_accuracy = copy.deepcopy(reference)
    another_accuracy["official_test_accuracy"] = 0.75
    assert _comparison(reference, another_accuracy)["numerical_equivalence_status"] == "difference_observed"
