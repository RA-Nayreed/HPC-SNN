import copy

import torch
from torch import nn
from torch.utils.data import Dataset

from fedapfa.configuration import load_federated_config
from fedapfa.datasets.dirichlet_partition import DirichletPartition
from fedapfa.federated.randomness import resolved_seeds
from fedapfa.training.federated import train_federated
from fedapfa.utilities.run_records import initialize_run
from fedapfa.utilities.serialization import sha256_json


class ImageDataset(Dataset):
    fedapfa_batch_kind = "image"

    def __init__(self, seed: int, count: int = 4):
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(count, 3, 4, 4, generator=generator)
        self.labels = torch.arange(count) % 2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3 * 4 * 4, 2)

    def reset_state(self):
        return None

    def forward(self, images, *, generator):
        torch.rand((), generator=generator, device=images.device)
        return self.linear(images.flatten(1)), {"image": images.mean()}


class ImageBundle:
    def __init__(self, config):
        self.resolved_seed_values = resolved_seeds(config)
        self.datasets = {"client_00": ImageDataset(1), "client_01": ImageDataset(2)}
        clients = [
            {
                "client_id": name,
                "indices": list(range(index * 4, index * 4 + 4)),
                "size": 4,
                "class_counts": {"0": 2, "1": 2},
            }
            for index, name in enumerate(self.datasets)
        ]
        artifact = {
            "client_count": 2,
            "clients": clients,
            "integrity_checks": {
                "complete_assignment": True,
                "unique_assignment": True,
                "minimum_size_satisfied": True,
                "validation_indices_excluded": True,
                "official_test_indices_excluded": True,
            },
        }
        artifact["partition_id"] = sha256_json(artifact)
        self.partition = DirichletPartition(artifact["partition_id"], artifact)
        self.split_artifact = {
            "dataset_identity": {"name": "synthetic-cifar10", "sha256": "fixture"},
            "training_indices": list(range(8)),
            "validation_indices": list(range(8, 12)),
        }
        self.split_artifact["split_id"] = sha256_json(self.split_artifact)
        self.validation_dataset = ImageDataset(3)

    def client_dataset(self, client_id):
        return self.datasets[client_id]


def test_one_reduced_sample_cifar10_federated_round(tmp_path):
    config = copy.deepcopy(
        load_federated_config(
            "experiments/published_fedsnn/cifar10/svgg9_bntt_dirichlet_alpha_0_5.yaml"
        )
    )
    config["name"] = "synthetic_cifar10_round"
    config["device"] = "cpu"
    config["output_root"] = str(tmp_path / "runs")
    config["federated"].update(
        {
            "rounds": 2,
            "clients": 2,
            "clients_per_round": 1,
            "participation_fraction": 0.5,
            "local_epochs": 1,
            "local_batch_size": 2,
            "data_loader_workers": 0,
            "persistent_workers": False,
        }
    )
    config["federated"]["partition"]["minimum_examples_per_client"] = 4
    bundle = ImageBundle(config)
    run = initialize_run(
        config,
        {
            "train": bundle.split_artifact["training_indices"],
            "validation": bundle.split_artifact["validation_indices"],
        },
        "synthetic CIFAR-10 federated verification",
    )
    result = train_federated(ImageClassifier(), bundle, config, run, stop_after_round=1)
    assert not result["completed"]
    assert result["completed_rounds"] == 1
    assert (run / "round_metrics.jsonl").read_text(encoding="utf-8").count("\n") == 1
