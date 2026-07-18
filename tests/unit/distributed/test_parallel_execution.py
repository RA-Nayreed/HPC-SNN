import copy
import os
import shutil
from collections import Counter

import pytest
import torch
from torch import nn

from fedapfa.configuration import (
    ConfigurationError,
    distributed_execution_identity,
    distributed_scientific_identity,
    load_device_capacity_manifest,
    load_distributed_evaluation_config,
    load_distributed_evaluation_manifest,
    process_device_mapping,
    validate_distributed_evaluation_config,
)
from fedapfa.distributed.assignment_broadcast import assign_clients, assignments_for_rank
from fedapfa.federated.checkpointing import (
    configuration_identity,
    load_federated_checkpoint,
    save_federated_checkpoint,
    state_identity,
)
from fedapfa.training.distributed_federated import (
    _gpu_utilization_record,
    _required_gpu_utilization_record,
    _selected_checkpoint_round,
)
from fedapfa.utilities.run_records import initialize_run

MANIFEST = "experiments/distributed_evaluation/manifest.yaml"
CAPACITY_MANIFEST = "experiments/device_capacity_evaluation/manifest.yaml"
CONFIG = "experiments/distributed_evaluation/shd/lif_fedavg_1_gpu.yaml"
CIFAR_CONFIG = "experiments/distributed_evaluation/cifar10/svgg9_bntt_noniid_2_gpu.yaml"


def test_manifest_expands_24_workload_paired_tasks():
    tasks = load_distributed_evaluation_manifest(MANIFEST)
    assert len(tasks) == 24
    assert sorted({task.seed for task in tasks}) == [7, 17, 27]
    assert Counter(task.dataset for task in tasks) == {"shd": 9, "ssc": 9, "cifar10": 6}
    for dataset in ("shd", "ssc", "cifar10"):
        templates = [task.config for task in tasks if task.dataset == dataset and task.seed == 7]
        assert len({repr(distributed_scientific_identity(value)) for value in templates}) == 1
        assert len({repr(distributed_execution_identity(value)) for value in templates}) == len(templates)
    identities = {
        dataset: repr(distributed_scientific_identity(next(task.config for task in tasks if task.dataset == dataset)))
        for dataset in ("shd", "ssc", "cifar10")
    }
    assert len(set(identities.values())) == 3
    for task in tasks:
        federation = task.config["federated"]
        if task.dataset in {"shd", "ssc"}:
            assert federation["aggregation_weighting"] == "example_count"
            assert federation["checkpoint_selection"] == "best_validation"
        else:
            assert federation["aggregation_weighting"] == "uniform"
            assert federation["checkpoint_selection"] == "final_round"
            assert task.config["parallel_execution"]["device_count"] <= 2


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("node_count", 2, "node_count 1"),
        ("device_count", 3, "1, 2, or 4"),
        ("process_count", 2, "device_count times"),
        ("control_backend", "gloo", "requires NCCL"),
        ("client_assignment", "rank_order", "assignment"),
        ("aggregation_order", "process_rank_order", "aggregation order"),
        ("synchronize_each_round", False, "synchronous"),
    ],
)
def test_parallel_configuration_rejects_incompatible_values(key, value, message):
    config = load_distributed_evaluation_config(CONFIG)
    changed = copy.deepcopy(config)
    changed["parallel_execution"][key] = value
    with pytest.raises(ConfigurationError, match=message):
        validate_distributed_evaluation_config(changed)


def test_parallel_configuration_rejects_scientific_drift_and_excess_devices():
    config = load_distributed_evaluation_config(CONFIG)
    changed = copy.deepcopy(config)
    changed["federated"]["learning_rate"] = 0.002
    with pytest.raises(ConfigurationError, match="learning rate"):
        validate_distributed_evaluation_config(changed)
    cifar = load_distributed_evaluation_config(CIFAR_CONFIG)
    cifar["parallel_execution"]["device_count"] = 4
    cifar["parallel_execution"]["process_count"] = 4
    with pytest.raises(ConfigurationError, match="device count cannot exceed selected clients"):
        validate_distributed_evaluation_config(cifar)


@pytest.mark.parametrize(
    ("world_size", "expected_counts"),
    [(1, [10]), (2, [5, 5]), (4, [3, 3, 2, 2])],
)
def test_selected_order_round_robin_assignment(world_size, expected_counts):
    clients = [f"client_{index:02d}" for index in range(10)]
    assignments = assign_clients(clients, world_size)
    assert [value.client_id for value in assignments] == clients
    assert [value.process_rank for value in assignments] == [index % world_size for index in range(10)]
    assert [len(assignments_for_rank(assignments, rank)) for rank in range(world_size)] == expected_counts
    assert len({value.client_id for value in assignments}) == 10


def test_assignment_and_process_device_mapping_cover_mps_topologies():
    with pytest.raises(ValueError, match="unique"):
        assign_clients(["client_00", "client_00"], 2)
    with pytest.raises(ValueError, match="positive"):
        assign_clients(["client_00"], 0)
    assert process_device_mapping(1, 1) == [{"process_rank": 0, "device_index": 0, "device_slot": 0}]
    assert [value["device_index"] for value in process_device_mapping(1, 4)] == [0, 0, 0, 0]
    assert [value["device_slot"] for value in process_device_mapping(1, 4)] == [0, 1, 2, 3]
    assert process_device_mapping(2, 4) == [
        {"process_rank": 0, "device_index": 0, "device_slot": 0},
        {"process_rank": 1, "device_index": 1, "device_slot": 0},
        {"process_rank": 2, "device_index": 0, "device_slot": 1},
        {"process_rank": 3, "device_index": 1, "device_slot": 1},
    ]
    config = load_distributed_evaluation_config(CONFIG)
    assert distributed_execution_identity(config)["process_to_device_mapping"] == [
        {"process_rank": 0, "device_index": 0, "device_slot": 0}
    ]


def test_two_physical_devices_with_two_mps_processes_each_validate():
    config = load_distributed_evaluation_config(
        "experiments/distributed_evaluation/shd/lif_fedavg_2_gpu.yaml"
    )
    config["parallel_execution"].update(
        {
            "client_processes_per_device": 2,
            "process_count": 4,
            "control_backend": "gloo",
            "cuda_process_service": "mps",
        }
    )
    validate_distributed_evaluation_config(config)
    assert distributed_execution_identity(config)["process_to_device_mapping"] == [
        {"process_rank": 0, "device_index": 0, "device_slot": 0},
        {"process_rank": 1, "device_index": 1, "device_slot": 0},
        {"process_rank": 2, "device_index": 0, "device_slot": 1},
        {"process_rank": 3, "device_index": 1, "device_slot": 1},
    ]


def test_cifar_validation_free_policy_selects_round_100():
    config = load_distributed_evaluation_config(CIFAR_CONFIG)
    assert config["dataset"]["validation_fraction"] == 0.0
    assert config["federated"]["checkpoint_selection"] == "final_round"
    assert _selected_checkpoint_round(
        config["federated"]["checkpoint_selection"],
        config["federated"]["rounds"],
        None,
    ) == 100


def test_mps_capacity_topologies_require_gloo_and_separate_execution_identity():
    tasks = load_device_capacity_manifest(CAPACITY_MANIFEST)
    assert len(tasks) == 9
    assert [task.config["parallel_execution"]["process_count"] for task in tasks] == [1, 1, 1, 2, 2, 2, 4, 4, 4]
    packed = [task for task in tasks if task.config["parallel_execution"]["process_count"] > 1]
    assert all(task.config["parallel_execution"]["control_backend"] == "gloo" for task in packed)
    assert all(task.config["parallel_execution"]["cuda_process_service"] == "mps" for task in packed)
    assert len({repr(distributed_execution_identity(task.config)) for task in tasks if task.seed == 7}) == 3
    changed = copy.deepcopy(packed[0].config)
    changed["parallel_execution"]["control_backend"] = "nccl"
    with pytest.raises(ConfigurationError, match="Gloo coordination"):
        validate_distributed_evaluation_config(changed)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("client_processes_per_device", 3, "must be 1, 2, or 4"),
        ("record_device_utilization", False, "must record physical-device utilization"),
        ("utilization_interval_seconds", 1, "interval must be 2 seconds"),
    ],
)
def test_parallel_measurement_and_process_capacity_are_strict(key, value, message):
    config = load_distributed_evaluation_config(CONFIG)
    if key in config["parallel_execution"]:
        config["parallel_execution"][key] = value
    else:
        config["execution_measurement"][key] = value
    with pytest.raises(ConfigurationError, match=message):
        validate_distributed_evaluation_config(config)


@pytest.mark.cuda
def test_cuda_mps_capability_requires_an_active_service_environment():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if shutil.which("nvidia-cuda-mps-control") is None:
        pytest.skip("CUDA MPS control is unavailable")
    if not os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
        pytest.skip("CUDA MPS service environment is not active")
    assert torch.cuda.device_count() >= 1


def test_gpu_utilization_record_uses_physical_device_samples(tmp_path):
    telemetry = tmp_path / "gpu.csv"
    telemetry.write_text(
        "timestamp, index, uuid, name, utilization.gpu, utilization.memory\n"
        "2026/01/01 00:00:00, 0, uuid0, GH200, 25, 1\n"
        "2026/01/01 00:00:02, 1, uuid1, GH200, 75, 1\n",
        encoding="utf-8",
    )
    record = _gpu_utilization_record(str(telemetry))
    assert record["sample_count"] == 2
    assert record["mean_percent"] == 50.0
    assert record["by_device_index"]["0"]["mean_percent"] == 25.0
    assert record["by_device_index"]["1"]["mean_percent"] == 75.0
    config = load_distributed_evaluation_config(
        "experiments/distributed_evaluation/shd/lif_fedavg_2_gpu.yaml"
    )
    required = _required_gpu_utilization_record(config, str(telemetry))
    assert required["sampling_interval_seconds"] == 2

    telemetry.write_text(
        "timestamp, index, uuid, name, utilization.gpu\n"
        "2026/01/01 00:00:00, 0, uuid0, GH200, 25\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="do not cover"):
        _required_gpu_utilization_record(config, str(telemetry))


def test_checkpoint_refuses_another_process_topology(tmp_path):
    config = load_distributed_evaluation_config(CONFIG)
    config["output_root"] = str(tmp_path)
    model = nn.Linear(2, 2)
    initialization_id = state_identity(model.state_dict())
    run_dir = initialize_run(config, {}, "distributed checkpoint test")
    checkpoint_path = run_dir / "checkpoints" / "last.pt"
    save_federated_checkpoint(
        checkpoint_path,
        model,
        config,
        run_dir,
        next_round=2,
        best_validation_accuracy=0.5,
        best_validation_round=1,
        selection_state={"generator_state": torch.Generator().manual_seed(3).get_state()},
        split_id="split",
        partition_id="partition",
        model_initialization_id=initialization_id,
        cumulative_download_bytes=1,
        cumulative_upload_bytes=1,
        client_records=[],
        round_records=[],
    )
    changed = copy.deepcopy(config)
    changed["parallel_execution"].update(
        {
            "client_processes_per_device": 2,
            "process_count": 2,
            "control_backend": "gloo",
            "cuda_process_service": "mps",
        }
    )
    changed["name"] = "shd_lif_fedavg_1_gpu_2_client_processes"
    changed["metadata"]["experiment"] = changed["name"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["configuration_id"] = configuration_identity(changed)
    checkpoint["resolved_config"] = changed
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(RuntimeError, match="parallel_execution"):
        load_federated_checkpoint(
            checkpoint_path, model, changed, run_dir, "split", "partition", initialization_id
        )
