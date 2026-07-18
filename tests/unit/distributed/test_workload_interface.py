import copy

import numpy as np
import pytest
import torch
from conftest import write_event_h5

from fedapfa.configuration import load_distributed_evaluation_config
from fedapfa.datasets.validation import DatasetValidationError
from fedapfa.federated.data_protocol import (
    FederatedWorkload,
    OfficialTestAccessError,
    prepare_federated_workload,
)
from fedapfa.federated.workload import prepare_federated_execution_workload


def _event_config(path, tmp_path, train_labels, validation_labels, test_labels):
    config = copy.deepcopy(load_distributed_evaluation_config(path))
    train = write_event_h5(tmp_path / "train.h5", train_labels)
    test = write_event_h5(tmp_path / "test.h5", test_labels)
    config["dataset"]["root"] = str(tmp_path)
    config["dataset"]["train_file"] = train.name
    config["dataset"]["test_file"] = test.name
    if validation_labels is not None:
        validation = write_event_h5(tmp_path / "validation.h5", validation_labels)
        config["dataset"]["validation_file"] = validation.name
    return config


def test_shd_workload_satisfies_common_interface(tmp_path):
    config = _event_config(
        "experiments/distributed_evaluation/shd/lif_fedavg_1_gpu.yaml",
        tmp_path,
        tuple(range(20)) * 100,
        None,
        tuple(range(20)) * 2,
    )
    bundle = prepare_federated_workload(config, coordinator=True)
    assert isinstance(bundle, FederatedWorkload)
    assert bundle.validation_dataset is not None
    assert sum(len(value) for value in bundle.partition.client_indices.values()) == len(bundle.train_indices)
    with pytest.raises(OfficialTestAccessError, match="before global model selection"):
        bundle.official_test_dataset(model_selected=False)


def test_execution_workload_provides_model_and_client_training_services(tmp_path):
    config = _event_config(
        "experiments/distributed_evaluation/shd/lif_fedavg_1_gpu.yaml",
        tmp_path,
        tuple(range(20)) * 100,
        None,
        tuple(range(20)) * 2,
    )
    workload = prepare_federated_execution_workload(config, coordinator=True)
    model = workload.model_factory(config)
    assert isinstance(workload.data, FederatedWorkload)
    assert callable(workload.client_training)
    assert model.__class__.__name__ == "AudioLIFSNN"


def test_ssc_workload_partitions_only_training_and_defers_test(tmp_path):
    config = _event_config(
        "experiments/distributed_evaluation/ssc/lif_128_fedavg_1_gpu.yaml",
        tmp_path,
        tuple(range(35)) * 60,
        tuple(range(35)) * 3,
        tuple(range(35)) * 4,
    )
    bundle = prepare_federated_workload(config, coordinator=True)
    assigned = [index for values in bundle.partition.client_indices.values() for index in values]
    assert sorted(assigned) == list(range(len(bundle.train_indices)))
    assert len(assigned) == len(set(assigned))
    assert bundle.split_artifact["validation_collection"] == "official_validation"
    assert bundle.validation_dataset.path.name == "validation.h5"
    assert bundle.official_test_access_count == 0
    with pytest.raises(OfficialTestAccessError, match="before global model selection"):
        bundle.official_test_dataset(model_selected=False)
    test = bundle.official_test_dataset(model_selected=True)
    assert test.path.name == "test.h5"
    assert bundle.official_test_access_count == 1


def test_noncoordinator_ssc_constructs_no_validation_or_test_dataset(tmp_path):
    config = _event_config(
        "experiments/distributed_evaluation/ssc/lif_128_fedavg_1_gpu.yaml",
        tmp_path,
        tuple(range(35)) * 60,
        tuple(range(35)) * 3,
        tuple(range(35)) * 4,
    )
    bundle = prepare_federated_workload(config, coordinator=False)
    assert isinstance(bundle, FederatedWorkload)
    assert bundle.validation_dataset is None
    assert bundle.official_test_access_count == 0


def test_ssc_official_collection_size_is_verified_before_partitioning(tmp_path):
    config = copy.deepcopy(
        load_distributed_evaluation_config(
            "experiments/distributed_evaluation/ssc/lif_128_fedavg_1_gpu.yaml"
        )
    )
    config["dataset"]["root"] = str(tmp_path)
    write_event_h5(tmp_path / "ssc_train.h5", tuple(range(35)) * 10)
    write_event_h5(tmp_path / "ssc_valid.h5", tuple(range(35)) * 3)
    write_event_h5(tmp_path / "ssc_test.h5", tuple(range(35)) * 4)
    with pytest.raises(DatasetValidationError, match="expected 75466 examples"):
        prepare_federated_workload(config, coordinator=True)


def test_cifar_workload_is_routed_through_common_interface(monkeypatch):
    config = load_distributed_evaluation_config(
        "experiments/distributed_evaluation/cifar10/svgg9_bntt_noniid_1_gpu.yaml"
    )
    marker = object()
    calls = []

    def prepare(config_value, *, construct_validation):
        calls.append((config_value["dataset"]["name"], construct_validation))
        return marker

    monkeypatch.setattr("fedapfa.datasets.cifar10.prepare_federated_cifar10", prepare)
    assert prepare_federated_workload(config, coordinator=False) is marker
    assert calls == [("cifar10", False)]


def test_cifar_bundle_satisfies_common_interface_without_validation(monkeypatch):
    from fedapfa.datasets import cifar10 as cifar_module

    class SyntheticCIFARTrainingData:
        def __init__(self, root):
            self.root = root
            self.targets = list(range(10)) * 100

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return torch.zeros(3, 32, 32), self.targets[index]

    monkeypatch.setattr(cifar_module, "_torchvision", lambda: (None, None, None))
    monkeypatch.setattr(cifar_module, "CIFAR10TrainingData", SyntheticCIFARTrainingData)
    monkeypatch.setattr(
        cifar_module,
        "cifar10_training_identity",
        lambda root: {"name": "cifar10_training", "examples": 1000},
    )
    config = load_distributed_evaluation_config(
        "experiments/distributed_evaluation/cifar10/svgg9_bntt_noniid_1_gpu.yaml"
    )
    bundle = prepare_federated_workload(config, coordinator=True)
    assigned = np.concatenate(list(bundle.partition.client_indices.values()))
    assert isinstance(bundle, FederatedWorkload)
    assert bundle.validation_dataset is None
    assert bundle.checkpoint_selection == "final_round"
    assert bundle.aggregation_weighting == "uniform"
    assert sorted(assigned.tolist()) == list(range(1000))
    assert len(np.unique(assigned)) == 1000
    assert bundle.official_test_access_count == 0
