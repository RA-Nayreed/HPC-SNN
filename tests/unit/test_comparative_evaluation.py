from __future__ import annotations

import copy
from pathlib import Path

import pytest

from fedapfa.configuration import (
    COMPARATIVE_SEEDS,
    NON_IID_EXECUTION_ORDERS,
    NON_IID_TREATMENTS,
    SCALING_TOPOLOGIES,
    ConfigurationError,
    comparative_scientific_identity,
    load_comparative_allocations,
    load_comparative_evaluation_manifest,
    validate_comparative_path_disjointness,
    validate_resolved_comparative_manifest,
    validate_resolved_comparative_pair,
)

ROOT = Path(__file__).resolve().parents[2]
SCALING = ROOT / "experiments/system_scaling_energy_evaluation/manifest.yaml"
NON_IID = ROOT / "experiments/non_iid_energy_evaluation/manifest.yaml"


def test_comparative_manifests_have_exact_disjoint_matrices() -> None:
    scaling = load_comparative_evaluation_manifest(SCALING)
    non_iid = load_comparative_evaluation_manifest(NON_IID)
    assert len(scaling) == len(non_iid) == 24
    assert {task.seed for task in (*scaling, *non_iid)} == set(COMPARATIVE_SEEDS) == {37, 47, 57}
    assert {task.config["comparative_evaluation"]["treatment_id"] for task in scaling} == set(SCALING_TOPOLOGIES)
    assert {task.config["comparative_evaluation"]["treatment_id"] for task in non_iid} == set(NON_IID_TREATMENTS)
    paths = [(task.config["output_root"], task.experiment, task.seed) for task in (*scaling, *non_iid)]
    assert len(paths) == len(set(paths)) == 48
    validate_comparative_path_disjointness(SCALING, NON_IID)


def test_scaling_topologies_and_fixed_execution_policy_are_exact() -> None:
    tasks = load_comparative_evaluation_manifest(SCALING)
    observed = {}
    for task in tasks:
        treatment = task.config["comparative_evaluation"]["treatment_id"]
        parallel = task.config["parallel_execution"]
        observed[treatment] = (
            parallel["node_count"],
            parallel["devices_per_node"],
            parallel["device_count"],
            parallel["process_count"],
        )
        assert parallel["client_processes_per_device"] == 1
        assert parallel["cuda_process_service"] == "none"
        assert parallel["control_backend"] == "nccl"
        assert task.config["scheduler"]["strategy"] == "example_count_longest_processing_time"
        assert task.config["aggregation_execution"]["topology"] == "flat_ordered"
        assert task.config["federated"]["partition"]["alpha"] == 0.5
    assert observed == SCALING_TOPOLOGIES


def test_non_iid_treatments_and_rotated_orders_are_exact() -> None:
    tasks = load_comparative_evaluation_manifest(NON_IID)
    for task in tasks:
        treatment = task.config["comparative_evaluation"]["treatment_id"]
        partition = task.config["federated"]["partition"]
        assert (partition["method"], partition["alpha"]) == NON_IID_TREATMENTS[treatment]
        parallel = task.config["parallel_execution"]
        assert (
            parallel["node_count"],
            parallel["devices_per_node"],
            parallel["device_count"],
            parallel["process_count"],
        ) == (1, 4, 4, 4)
    allocations = load_comparative_allocations(NON_IID)
    assert len(allocations) == 6
    assert all(allocation.execution_order == NON_IID_EXECUTION_ORDERS[allocation.seed] for allocation in allocations)
    assert all(len(allocation.tasks) == 4 for allocation in allocations)


def test_scaling_has_one_execution_per_allocation() -> None:
    allocations = load_comparative_allocations(SCALING)
    assert len(allocations) == 24
    assert all(len(allocation.tasks) == len(allocation.execution_order) == 1 for allocation in allocations)


def test_resolved_whitelist_rejects_unintended_difference() -> None:
    tasks = load_comparative_evaluation_manifest(SCALING)
    reference = next(
        task.config
        for task in tasks
        if task.dataset == "shd"
        and task.seed == 37
        and task.config["comparative_evaluation"]["treatment_id"] == "one_node_one_gpu"
    )
    candidate = copy.deepcopy(reference)
    candidate["federated"]["learning_rate"] = 0.002
    with pytest.raises(ConfigurationError, match="outside the whitelist"):
        validate_resolved_comparative_pair(reference, candidate, "system_scaling_energy_evaluation")


def test_resolved_manifest_identity_is_checked_after_path_overrides() -> None:
    scaling = validate_resolved_comparative_manifest(
        SCALING,
        data_root="/tmp/comparative-data",
        output_root="/tmp/runs/system_scaling_energy_evaluation",
    )
    non_iid = validate_resolved_comparative_manifest(
        NON_IID,
        data_root="/tmp/comparative-data",
        output_root="/tmp/runs/non_iid_energy_evaluation",
    )
    assert len(scaling) == len(non_iid) == 24
    assert {(value["dataset"], value["seed"]) for value in scaling} == {
        (dataset, seed) for dataset in ("shd", "ssc") for seed in COMPARATIVE_SEEDS
    }


def test_scaling_scientific_identity_is_topology_independent() -> None:
    tasks = load_comparative_evaluation_manifest(SCALING)
    selected = [task for task in tasks if task.dataset == "shd" and task.seed == 37]
    identities = [comparative_scientific_identity(task.config) for task in selected]
    assert all(value == identities[0] for value in identities[1:])
    assert identities[0]["seed_streams"]["client_training"] == selected[0].config["seed_streams"]["client_training"]


def test_non_iid_whitelist_allows_only_distribution_identity() -> None:
    tasks = load_comparative_evaluation_manifest(NON_IID)
    iid = next(
        task.config
        for task in tasks
        if task.dataset == "ssc" and task.seed == 37 and task.config["comparative_evaluation"]["treatment_id"] == "iid"
    )
    alpha = next(
        task.config
        for task in tasks
        if task.dataset == "ssc"
        and task.seed == 37
        and task.config["comparative_evaluation"]["treatment_id"] == "dirichlet_alpha_0_1"
    )
    record = validate_resolved_comparative_pair(iid, alpha, "non_iid_energy_evaluation")
    assert record["whitelist_version"] == "resolved_leaf_paths_v1"
    assert set(record["observed_difference_paths"]) == {
        "name",
        "metadata.experiment",
        "comparative_evaluation.treatment_id",
        "federated.partition.method",
        "federated.partition.alpha",
    }


def test_comparative_slurm_launchers_encode_required_policy() -> None:
    scaling = (ROOT / "scripts/slurm/system_scaling_energy.sbatch").read_text(encoding="utf-8")
    non_iid = (ROOT / "scripts/slurm/non_iid_energy.sbatch").read_text(encoding="utf-8")
    wrapper = (ROOT / "scripts/slurm/submit_roihu_system_scaling_energy.sh").read_text(encoding="utf-8")
    calibration = (ROOT / "scripts/slurm/comparative_measurement_calibration.sbatch").read_text(encoding="utf-8")
    for text in (scaling, non_iid):
        assert "gpumedium" in text
        assert '"${python_bin}" -m torch.distributed.run' in text
        assert "TORCH_NCCL_ASYNC_ERROR_HANDLING=1" in text
        assert "export NCCL_ASYNC_ERROR_HANDLING" not in text
        assert "mpirun" not in text and "mpiexec" not in text
        assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=" not in text
    assert "--rdzv-backend=c10d" in scaling
    assert "--nnodes=2 --nproc-per-node=2" in scaling
    assert "217086M 434172M 868344M 434172M" in wrapper
    assert "%1" not in wrapper
    assert "calibrate_comparative_measurement" in calibration
    assert "--rdzv-backend=c10d" in calibration
    assert "gpumedium" in calibration
