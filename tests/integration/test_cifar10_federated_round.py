import copy
import json

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
            "validation_indices": [],
        }
        self.split_artifact["split_id"] = sha256_json(self.split_artifact)
        self.validation_dataset = None
        self.test_dataset = ImageDataset(3)
        self.official_test_access_count = 0

    def client_dataset(self, client_id):
        return self.datasets[client_id]

    def official_test_dataset(self, model_selected):
        if not model_selected:
            raise RuntimeError("official test access requires model selection")
        if self.official_test_access_count:
            raise RuntimeError("official test access is permitted once")
        self.official_test_access_count += 1
        return self.test_dataset


def _model():
    torch.manual_seed(19)
    return ImageClassifier()


def _config(tmp_path):
    config = copy.deepcopy(
        load_federated_config("experiments/published_fedsnn/cifar10/paper_reported_iid_evaluation.yaml")
    )
    config["name"] = "synthetic_cifar10_final_round"
    config["metadata"]["experiment"] = config["name"]
    config["device"] = "cpu"
    config["output_root"] = str(tmp_path / "runs")
    config["acceptance"]["expected_model_class"] = "ImageClassifier"
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
    return config


def _run(config, bundle):
    return initialize_run(
        config,
        {
            "train": bundle.split_artifact["training_indices"],
            "validation": bundle.split_artifact["validation_indices"],
        },
        "synthetic CIFAR-10 final-round verification",
    )


def test_zero_validation_partial_execution_does_not_validate_or_access_official_test(tmp_path, monkeypatch):
    config = _config(tmp_path)
    bundle = ImageBundle(config)
    run = _run(config, bundle)

    def reject_validation(*args, **kwargs):
        raise AssertionError("internal validation must not be called")

    monkeypatch.setattr("fedapfa.training.federated.validate_global_model", reject_validation)
    result = train_federated(_model(), bundle, config, run, stop_after_round=1)
    assert not result["completed"]
    assert result["completed_rounds"] == 1
    assert bundle.official_test_access_count == 0
    assert not (run / "checkpoints" / "best.pt").exists()
    assert (run / "round_metrics.jsonl").read_text(encoding="utf-8").count("\n") == 1
    client_record = json.loads((run / "client_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert client_record["client_population_examples"] == 4
    assert client_record["local_training_examples_presented"] == 4
    assert client_record["aggregation_weight"] == 1.0
    round_record = json.loads((run / "round_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert round_record["validation_accuracy"] is None
    assert round_record["current_best_validation_round"] is None


def test_zero_validation_final_round_resume_uses_last_checkpoint_and_one_test_access(tmp_path):
    config = _config(tmp_path)
    first_bundle = ImageBundle(config)
    run = _run(config, first_bundle)
    partial = train_federated(_model(), first_bundle, config, run, stop_after_round=1)
    assert not partial["completed"] and first_bundle.official_test_access_count == 0

    resumed_bundle = ImageBundle(config)
    result = train_federated(
        _model(),
        resumed_bundle,
        config,
        run,
        resume_checkpoint=run / "checkpoints" / "last.pt",
    )
    assert result["completed"]
    assert result["selected_round"] == 2
    assert result["selected_checkpoint_artifact"] == "checkpoints/last.pt"
    assert result["best_validation_accuracy"] is None
    assert result["final_validation_accuracy"] is None
    assert result["selected_validation"] is None
    assert resumed_bundle.official_test_access_count == 1
    assert not (run / "checkpoints" / "best.pt").exists()
    official = json.loads((run / "official_test_metrics.json").read_text(encoding="utf-8"))
    assert official["access_count"] == 1 and official["selected_round"] == 2
    checkpoint = torch.load(run / "checkpoints" / "last.pt", weights_only=False)
    assert checkpoint["best_validation_accuracy"] is None
    assert checkpoint["best_validation_round"] is None
