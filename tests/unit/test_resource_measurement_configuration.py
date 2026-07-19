import copy

import pytest

from fedapfa.configuration import (
    ConfigurationError,
    load_resource_measurement_config,
    load_resource_measurement_manifest,
    validate_resource_measurement_config,
)

MANIFEST = "experiments/resource_measurement/manifest.yaml"
SHD = "experiments/resource_measurement/shd/lif_client_resource.yaml"
SSC = "experiments/resource_measurement/ssc/lif_128_client_resource.yaml"


def test_manifest_has_exact_scientific_matrix():
    tasks = load_resource_measurement_manifest(MANIFEST)
    assert len(tasks) == 6
    assert [(task.dataset, task.seed) for task in tasks] == [
        ("shd", 7),
        ("shd", 17),
        ("shd", 27),
        ("ssc", 7),
        ("ssc", 17),
        ("ssc", 27),
    ]
    shd = load_resource_measurement_config(SHD)
    ssc = load_resource_measurement_config(SSC)
    assert shd["federated"]["local_batch_size"] == 32
    assert shd["model"]["hidden_dims"] == [256, 256]
    assert ssc["federated"]["local_batch_size"] == 256
    assert ssc["model"]["hidden_dims"] == [128, 128]
    assert ssc["dataset"]["validation_file"] == "ssc_valid.h5"


@pytest.mark.parametrize(
    ("path", "mutate"),
    [
        (SHD, lambda value: value["parallel_execution"].update(device_count=2)),
        (SHD, lambda value: value["parallel_execution"].update(process_count=2)),
        (SHD, lambda value: value["parallel_execution"].update(client_processes_per_device=2)),
        (SHD, lambda value: value["parallel_execution"].update(cuda_process_service="mps")),
        (SHD, lambda value: value["parallel_execution"].update(control_backend="gloo")),
        (SHD, lambda value: value.update(seed=5)),
        (SHD, lambda value: value["federated"].update(clients=21)),
        (SHD, lambda value: value["federated"].update(clients_per_round=9)),
        (SHD, lambda value: value["federated"].update(rounds=99)),
        (SHD, lambda value: value["federated"].update(local_batch_size=16)),
        (
            SHD,
            lambda value: value["federated"].update(
                official_test_evaluation_during_rounds=True
            ),
        ),
        (SHD, lambda value: value["training"].update(max_train_batches=1)),
        (SHD, lambda value: value["subset"].update(train_examples=1)),
        (SHD, lambda value: value["resource_measurement"].update(enabled=False)),
        (SHD, lambda value: value["resource_measurement"].update(sampling_backend="nvidia-smi")),
        (SHD, lambda value: value["resource_measurement"].update(sampling_interval_ms=200)),
        (SHD, lambda value: value["resource_measurement"].update(idle_before_seconds=29)),
        (SSC, lambda value: value["dataset"].update(validation_file=None)),
    ],
)
def test_incompatible_protocols_are_rejected(path, mutate):
    config = copy.deepcopy(load_resource_measurement_config(path))
    mutate(config)
    with pytest.raises(ConfigurationError):
        validate_resource_measurement_config(config)
