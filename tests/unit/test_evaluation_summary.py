import copy
import hashlib
import json
from dataclasses import replace
from datetime import datetime

import pytest
import torch

from fedapfa.analysis import evaluation_summary
from fedapfa.analysis.evaluation_figures import generate_evaluation_figures
from fedapfa.configuration import evaluation_scientific_identity, load_evaluation_manifest
from fedapfa.federated.aggregation import aggregation_tensor_policy
from fedapfa.federated.checkpointing import configuration_identity
from fedapfa.utilities.run_records import run_directory

SCHEDULING = "experiments/scheduling_evaluation/manifest.yaml"
HIERARCHY = "experiments/hierarchical_reduction_evaluation/manifest.yaml"
COMMIT = "1" * 40
GPU_UUIDS = tuple(f"{value:08x}-0000-4000-8000-000000000000" for value in range(1, 5))


def _tasks(manifest, root):
    resolved = []
    for task in load_evaluation_manifest(manifest):
        config = copy.deepcopy(task.config)
        config["output_root"] = str(root)
        config["federated"]["rounds"] = 1
        config["federated"]["clients"] = 2
        config["federated"]["clients_per_round"] = 2
        config["federated"]["participation_fraction"] = 1.0
        resolved.append(replace(task, config=config))
    return resolved


def _order(collection, seed):
    return evaluation_summary._expected_treatment_order(collection, seed)


def _runtime(collection, treatment):
    if collection == "scheduling_evaluation":
        return {
            "round_robin": 10.0,
            "example_count_longest_processing_time": 9.2,
            "event_structure_longest_processing_time": 9.0,
        }[treatment]
    return 10.0


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, values):
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values), encoding="utf-8")


def _write_run(task, root, collection, allocation_index):
    config = task.config
    treatment = (
        config["scheduler"]["strategy"]
        if collection == "scheduling_evaluation"
        else config["aggregation_execution"]["topology"]
    )
    path = root / run_directory(config).name
    (path / "checkpoints").mkdir(parents=True)
    runtime = _runtime(collection, treatment)
    hierarchy = collection == "hierarchical_reduction_evaluation" and treatment == "node_hierarchical"
    state = {"weight": torch.tensor([1.0 + (5e-7 if hierarchy else 0.0), 2.0])}
    torch.save({"global_model_state": state}, path / "checkpoints" / "best.pt")
    model_record = {
        "model_load_count": 1,
        "model_sha256": config["scheduler"]["cost_model"]["expected_sha256"],
        "row_provenance": {
            "total_accepted_rows": 6000,
            "model_fitting_rows": 4000,
            "untouched_evaluation_rows": 2000,
            "seed_27_overlap_with_fitting_or_selection_rows": 0,
            "seed_27_row_hash_in_coefficient_fitting_rows": False,
            "normalization_uses_seed_27": False,
            "coefficient_fitting_uses_seed_27": False,
            "regression_family_selection_uses_seed_27": False,
            "feature_selection_uses_seed_27": False,
            "hyperparameter_selection_uses_seed_27": False,
        },
    }
    clients = ["client_a", "client_b"]
    strategy = config["scheduler"]["strategy"]
    privacy_fields = (
        []
        if strategy == "round_robin"
        else ["example_count"]
        if strategy == "example_count_longest_processing_time"
        else config["scheduler"]["cost_model"]["feature_order"]
    )
    scheduler = {
        "strategy": strategy,
        "total_scheduler_seconds": runtime * 0.005,
        "model_sha256": model_record["model_sha256"],
        "model_provenance_identity": "model-provenance",
        "feature_availability": {
            name: strategy == "event_structure_longest_processing_time"
            for name in config["scheduler"]["cost_model"]["feature_order"]
        },
        "privacy_metadata": [
            {
                "client_id": client,
                "field": name,
                "contains_label_information": False,
                "raw_events_leave_client": False,
            }
            for client in clients
            for name in privacy_fields
        ],
    }
    assignments = [
        {
            "client_id": client,
            "selected_position": position,
            "process_rank": position,
            "assigned_node_rank": 0,
            "cost": 1.0 + position,
            "actual_client_wall_duration_seconds": 1.1 + position,
            "cost_source": {
                "round_robin": "selected_position",
                "example_count_longest_processing_time": "training_example_count",
                "event_structure_longest_processing_time": ("frozen_event_structure_wall_time_prediction"),
            }[config["scheduler"]["strategy"]],
            "features": (
                {name: float(position + 1) for name in config["scheduler"]["cost_model"]["feature_order"]}
                if config["scheduler"]["strategy"] == "event_structure_longest_processing_time"
                else None
            ),
        }
        for position, client in enumerate(clients)
    ]
    logical_inter = 0 if collection == "scheduling_evaluation" else 400 if hierarchy else 1000
    if collection == "hierarchical_reduction_evaluation":
        assignments[1]["assigned_node_rank"] = 1
    round_record = {
        "round_number": 1,
        "selected_client_ids": clients,
        "selected_client_order": clients,
        "client_assignments": assignments,
        "scheduler": scheduler,
        "total_round_time_seconds": runtime,
        "parallel_client_training_wall_time_seconds": runtime * 0.6,
        "aggregation_time_seconds": runtime * 0.1,
        "node_local_reduction_time_seconds": runtime * 0.02 if hierarchy else 0.0,
        "node_leader_synchronization_time_seconds": runtime * 0.01 if hierarchy else 0.0,
        "inter_node_contribution_movement_time_seconds": runtime * 0.01 if hierarchy else 0.0,
        "global_reduction_time_seconds": runtime * 0.04 if hierarchy else runtime * 0.1,
        "logical_intra_node_bytes": 200,
        "logical_inter_node_bytes": logical_inter,
        "predicted_logical_inter_node_bytes": logical_inter,
        "client_result_collection_bytes": 200 + logical_inter,
        "model_distribution_bytes": 3000,
        "model_sized_payloads_crossing_node_boundaries": (0 if collection == "scheduling_evaluation" else 1),
        "predicted_load_imbalance": 0.1,
        "process_load_imbalance": 0.08,
        "process_busy_time_seconds": {"0": 6.0, "1": 5.8, "2": 5.6, "3": 5.5},
        "estimated_idle_time_seconds_by_process_rank": {
            "0": 0.0,
            "1": 0.2,
            "2": 0.4,
            "3": 0.5,
        },
        "scheduler_overhead_fraction": 0.005,
        "client_example_counts": [2, 3],
        "client_training_examples_presented": [2, 3],
        "ordered_update_identities": ["update-a", "update-b"],
        "global_model_identity_after_aggregation": "aggregate-identity",
        "aggregation_weights": [0.4, 0.6],
        "aggregation_order": "selected_client_order",
        "validation_loss": 0.5,
        "validation_accuracy": 0.75,
        "validation_macro_f1": 0.74,
        "selected_checkpoint": True,
    }
    client_records = [
        {
            "round_number": 1,
            "client_id": client,
            "resolved_training_seed": 100 + position,
        }
        for position, client in enumerate(clients)
    ]
    process_mapping = [
        {
            "rank": rank,
            "host": f"node-{rank // 2}" if collection == "hierarchical_reduction_evaluation" else "node-0",
            "gpu_uuid": GPU_UUIDS[rank],
            "node_rank": rank // (2 if collection == "hierarchical_reduction_evaluation" else 4),
            "local_rank": rank % (2 if collection == "hierarchical_reduction_evaluation" else 4),
            "device_index": rank % (2 if collection == "hierarchical_reduction_evaluation" else 4),
        }
        for rank in range(4)
    ]
    expected_order = _order(collection, task.seed)
    treatment_position = expected_order.index(treatment)
    allocation_start = datetime.fromisoformat("2026-01-01T00:00:00").timestamp()
    treatment_start = allocation_start + 5.0 + sum(
        _runtime(collection, previous) for previous in expected_order[:treatment_position]
    ) + 2.0 * treatment_position
    treatment_end = treatment_start + runtime
    resource = {
        "job_id": f"900_{allocation_index}",
        "array_job_id": "900",
        "array_task_id": str(allocation_index),
        "partition": "gpumedium",
        "allocated_gpus_on_node": "4" if collection == "scheduling_evaluation" else "2",
        "allocated_nodes": "1" if collection == "scheduling_evaluation" else "2",
        "node_list": "node-0" if collection == "scheduling_evaluation" else "node-[0-1]",
        "cpus_per_task": "288" if collection == "scheduling_evaluation" else "144",
        "gpu_telemetry_path": "/scratch/telemetry.csv",
        "mps_log_archive": None,
        "mps_active_thread_percentage": None,
        "within_allocation_execution_order": ",".join(expected_order),
        "treatment_position": str(treatment_position + 1),
        "allocated_gpu_uuids": ",".join(f"GPU-{value}" for value in GPU_UUIDS),
    }
    measurements = {
        "completed": True,
        "rounds": [
            {
                "round_number": 1,
                "peak_cuda_memory_bytes_by_process_rank": {str(rank): 100 + rank for rank in range(4)},
                "peak_cuda_reserved_bytes_by_process_rank": {str(rank): 200 + rank for rank in range(4)},
            }
        ],
        "gpu_utilization": {"mean_percent": 50.0},
        "resource_allocation": resource,
        "process_mapping": process_mapping,
        "execution_start_unix_seconds": treatment_start,
        "execution_end_unix_seconds": treatment_end,
    }
    scientific_identity = evaluation_scientific_identity(config)
    final = {
        "completed": True,
        "configuration_id": configuration_identity(config),
        "scheduler_model": model_record,
        "selected_checkpoint_artifact": "checkpoints/best.pt",
        "selected_round": 1,
        "test": {"accuracy": 0.8, "macro_f1": 0.79, "predictions": [0, 1, 0]},
        "scheduler_overhead_fraction": 0.005,
        "internal_treatment_duration_seconds": runtime,
        "derived_treatment_gpu_exposure_hours": runtime * 4 / 3600,
        "aggregation_tensor_policy": aggregation_tensor_policy(),
        "scientific_identity": scientific_identity,
        "split_id": "split",
        "partition_id": "partition",
        "model_initialization_id": "initialization",
        "logical_communication": {"cumulative_total_bytes": 5000},
        "data_protocol": {"official_test_access_count": 1},
    }
    acceptance = {
        "completed": True,
        "execution_provenance": {"git_commit": COMMIT},
    }
    official = {"access_count": 1, "evaluation_completed": True}
    environment = {"cuda_runtime": "13.0", "nccl": [2, 27, 5]}
    git_record = {"commit": COMMIT, "dirty": False, "worktree_sha256": None}
    _write_json(path / "acceptance.json", acceptance)
    _write_json(path / "final_metrics.json", final)
    _write_json(path / "official_test_metrics.json", official)
    _write_json(path / "execution_measurements.json", measurements)
    _write_json(path / "environment.json", environment)
    _write_json(path / "git.json", git_record)
    _write_jsonl(path / "round_metrics.jsonl", [round_record])
    _write_jsonl(path / "client_metrics.jsonl", client_records)


def _fixture(tmp_path, manifest, collection, monkeypatch):
    root = tmp_path / collection
    output = tmp_path / f"{collection}_summary"
    root.mkdir()
    tasks = _tasks(manifest, root)
    allocation_by_pair = {
        (dataset, seed): index
        for index, (dataset, seed) in enumerate((dataset, seed) for dataset in ("shd", "ssc") for seed in (37, 47, 57))
    }
    for task in tasks:
        _write_run(task, root, collection, allocation_by_pair[(task.dataset, task.seed)])
    accounting = tmp_path / f"{collection}_accounting.txt"
    rows = ["JobIDRaw|State|ExitCode|ElapsedRaw|AllocTRES|Start|End\n"]
    for index in range(6):
        rows.append(f"900_{index}|COMPLETED|0:0|100|billing=1,gres/gpu=4|2026-01-01T00:00:00|2026-01-01T00:01:40\n")
    accounting.write_text("".join(rows), encoding="utf-8")
    monkeypatch.setattr(evaluation_summary, "load_evaluation_manifest", lambda _: tasks)
    summary = evaluation_summary.summarize_evaluation(
        manifest,
        root,
        output,
        slurm_accounting=accounting,
    )
    return summary, output


def test_scheduling_summary_calculations_and_adoption_boundaries(tmp_path, monkeypatch):
    summary, output = _fixture(tmp_path, SCHEDULING, "scheduling_evaluation", monkeypatch)
    assert summary["valid"] and summary["allocation_count"] == 6
    assert summary["decision"]["decision"] == "event_structure_scheduler_adopted"
    event_shd = next(
        value
        for value in summary["groups"]
        if value["dataset"] == "shd" and value["strategy"] == "event_structure_longest_processing_time"
    )
    assert event_shd["metrics"]["total_runtime_seconds"]["mean"] == 9.0
    assert event_shd["metrics"]["gpu_utilization_percent"]["mean"] == 50.0
    assert event_shd["metrics"]["paired_speedup"]["mean"] == pytest.approx(10 / 9)
    assert event_shd["metrics"]["paired_runtime_reduction"]["mean"] == pytest.approx(0.1)
    allocation = summary["allocations"][0]
    assert allocation["slurm_allocation_elapsed_seconds"] == 100
    assert allocation["slurm_allocation_gpu_hours"] == pytest.approx(100 * 4 / 3600)
    assert allocation["allocation_initialization_seconds"] == 5
    assert allocation["between_treatment_overhead_seconds"] == 4
    assert allocation["remaining_allocation_overhead_seconds"] == pytest.approx(62.8)
    assert allocation["allocation_reconciliation_error_seconds"] == pytest.approx(0, abs=1e-6)
    assert (
        sum(value["internal_treatment_duration_seconds"] for value in allocation["treatments"])
        + allocation["allocation_initialization_seconds"]
        + allocation["between_treatment_overhead_seconds"]
        + allocation["remaining_allocation_overhead_seconds"]
        + allocation["allocation_reconciliation_error_seconds"]
        == pytest.approx(allocation["slurm_allocation_elapsed_seconds"])
    )
    assert summary["total_slurm_allocation_gpu_hours"] == pytest.approx(6 * 100 * 4 / 3600)
    assert all("slurm_allocation_gpu_hours" not in value for value in summary["runs"])
    assert all("slurm_allocation_elapsed_seconds" not in value for value in summary["runs"])
    assert all(value["derived_treatment_gpu_exposure_hours"] > 0 for value in summary["runs"])
    assert (output / "scheduling_evaluation_summary.json").is_file()
    changed_pairs = copy.deepcopy(summary["paired_records"])
    next(value for value in changed_pairs if value["treatment"] == "event_structure_longest_processing_time")[
        "paired_runtime_reduction"
    ] = -0.03
    decision = evaluation_summary._decision("scheduling_evaluation", summary["runs"], changed_pairs, True)
    assert decision["decision"] == "event_structure_scheduler_not_adopted"
    assert decision["conditions"]["no_dataset_seed_pair_more_than_two_percent_slower"] is False
    assert decision["pairing_semantics"]["pairing_unit"] == "dataset_seed"
    assert decision["pairing_semantics"]["datasets_pooled"] is False
    assert len(decision["paired_dataset_seed_runtime_reductions"]) == 6
    assert decision["dataset_mean_paired_runtime_reductions"]["shd"] == pytest.approx(
        sum(
            value["paired_runtime_reduction"]
            for value in changed_pairs
            if value["dataset"] == "shd" and value["treatment"] == "event_structure_longest_processing_time"
        )
        / 3
    )
    assert decision["improved_seed_counts_by_dataset"]["shd"] == 2

    bad_record = {
        "job_id": "900_0",
        "start_unix_seconds": datetime.fromisoformat("2026-01-01T00:00:00").timestamp(),
        "end_unix_seconds": datetime.fromisoformat("2026-01-01T00:01:40").timestamp(),
        "slurm_allocation_elapsed_seconds": 90,
        "slurm_allocation_gpu_hours": 0.1,
        "allocated_gpu_count": 4,
        "state": "COMPLETED",
        "exit_code": "0:0",
        "allocation": "gres/gpu=4",
        "start": "2026-01-01T00:00:00",
        "end": "2026-01-01T00:01:40",
    }
    allocation_runs = [value for value in summary["runs"] if value["allocation_reconciliation_id"] == "900_0"]
    with pytest.raises(ValueError, match="reconciliation error exceeds"):
        evaluation_summary._reconcile_allocation(bad_record, allocation_runs, "scheduling_evaluation")


def test_hierarchical_summary_calculations_and_retention_boundaries(tmp_path, monkeypatch):
    summary, output = _fixture(tmp_path, HIERARCHY, "hierarchical_reduction_evaluation", monkeypatch)
    assert summary["valid"] and summary["allocation_count"] == 6
    assert summary["decision"]["decision"] == "node_hierarchical_reduction_retained"
    hierarchy_shd = next(
        value
        for value in summary["groups"]
        if value["dataset"] == "shd" and value["aggregation_topology"] == "node_hierarchical"
    )
    assert hierarchy_shd["metrics"]["logical_inter_node_bytes"]["mean"] == 400
    assert hierarchy_shd["metrics"]["paired_runtime_reduction"]["mean"] == 0
    assert hierarchy_shd["metrics"]["maximum_absolute_parameter_difference"]["mean"] > 0
    assert all(value["prediction_identity"] is True for value in hierarchy_shd["paired_numerical_classifications"])
    assert (output / "hierarchical_reduction_evaluation_summary.csv").is_file()
    changed_pairs = copy.deepcopy(summary["paired_records"])
    changed_pairs[0]["runtime_regression_fraction"] = 0.021
    decision = evaluation_summary._decision("hierarchical_reduction_evaluation", summary["runs"], changed_pairs, True)
    assert decision["decision"] == "node_hierarchical_reduction_not_retained"
    assert decision["conditions"]["no_material_runtime_regression"] is False


@pytest.mark.parametrize(
    ("manifest", "collection", "expected_count"),
    [
        (SCHEDULING, "scheduling_evaluation", 13),
        (HIERARCHY, "hierarchical_reduction_evaluation", 8),
    ],
)
def test_figure_generation_is_deterministic_and_reads_validated_summary(
    tmp_path, monkeypatch, manifest, collection, expected_count
):
    _, summary_output = _fixture(tmp_path, manifest, collection, monkeypatch)
    summary_path = summary_output / f"{collection}_summary.json"
    figure_output = tmp_path / f"{collection}_figures"
    first = generate_evaluation_figures(summary_path, figure_output)
    first_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in first}
    second = generate_evaluation_figures(summary_path, figure_output)
    second_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in second}
    assert len(first) == expected_count
    assert first_hashes == second_hashes
