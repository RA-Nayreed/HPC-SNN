import copy
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from fedapfa.configuration import load_federated_config
from fedapfa.datasets.dirichlet_partition import DirichletPartition
from fedapfa.federated.aggregation import clone_state_dict
from fedapfa.federated.checkpointing import load_federated_checkpoint, state_identity
from fedapfa.federated.client import evaluate_model, train_client
from fedapfa.federated.communication_accounting import model_payload_bytes
from fedapfa.federated.data_protocol import OfficialTestAccessError
from fedapfa.federated.fedavg import aggregate_client_results
from fedapfa.federated.randomness import derive_seed, resolved_seeds
from fedapfa.models.shd_lif import AudioLIFSNN
from fedapfa.training.federated import train_federated
from fedapfa.utilities.run_records import initialize_run, plan_run
from fedapfa.utilities.serialization import sha256_json


class SequenceDataset(Dataset):
    def __init__(self, offset, count=4):
        generator = torch.Generator().manual_seed(100 + offset)
        self.samples = [(torch.rand((3, 4), generator=generator), (offset + index) % 2) for index in range(count)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class SyntheticFederatedBundle:
    def __init__(self, config):
        self.config = config
        self.resolved_seed_values = resolved_seeds(config)
        self.datasets = {"client_00": SequenceDataset(0), "client_01": SequenceDataset(4)}
        clients = [
            {
                "client_id": client_id,
                "indices": list(range(index * 4, index * 4 + 4)),
                "size": 4,
                "class_counts": {"0": 2, "1": 2},
                "label_entropy_bits": 1.0,
            }
            for index, client_id in enumerate(self.datasets)
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
            "training_indices": list(range(8)),
            "validation_indices": list(range(8, 12)),
        }
        self.split_artifact["split_id"] = sha256_json(self.split_artifact)
        self.validation_dataset = SequenceDataset(8)
        self.test_dataset = SequenceDataset(12)
        self.official_test_access_count = 0

    def client_dataset(self, client_id):
        return self.datasets[client_id]

    def official_test_dataset(self, model_selected):
        if not model_selected:
            raise OfficialTestAccessError("official test access requires model selection")
        if self.official_test_access_count:
            raise OfficialTestAccessError("official test access is permitted once")
        self.official_test_access_count += 1
        return self.test_dataset


def _config(root: Path):
    config = load_federated_config(
        "experiments/federated_baselines/shd/lif_dirichlet_alpha_0_5_participation_0_50.yaml"
    )
    config = copy.deepcopy(config)
    config["name"] = "synthetic_fedavg_resumption"
    config["device"] = "cpu"
    config["output_root"] = str(root)
    config["federated"].update(
        {
            "rounds": 2,
            "clients": 2,
            "participation_fraction": 0.5,
            "clients_per_round": 1,
            "local_batch_size": 4,
            "data_loader_workers": 0,
            "persistent_workers": False,
        }
    )
    config["federated"]["partition"]["minimum_examples_per_client"] = 4
    return config


def _model(config):
    seed = resolved_seeds(config)["model_initialization"]
    torch.manual_seed(seed)
    return AudioLIFSNN(
        input_features=4,
        hidden_dims=[4, 4],
        classes=2,
        neuron=config["model"]["neuron"],
        attention=config["model"]["attention"],
        dropout=config["model"]["dropout"],
        batch_normalization=False,
        bias=True,
    )


def _new_run(config, bundle):
    return initialize_run(
        config,
        {"train": bundle.split_artifact["training_indices"], "validation": bundle.split_artifact["validation_indices"]},
        "synthetic federated verification",
    )


def test_client_isolation_state_reset_and_single_client_equivalence(tmp_path):
    config = _config(tmp_path / "runs")
    bundle = SyntheticFederatedBundle(config)
    server = _model(config)
    before = clone_state_dict(server.state_dict())
    result = train_client(
        server,
        bundle.client_dataset("client_00"),
        "client_00",
        1,
        config,
        torch.device("cpu"),
        derive_seed(config["seed"], config["seed_streams"]["client_training"], 1, "client_00"),
        model_payload_bytes(server.state_dict()),
    )
    assert all(torch.equal(before[name], server.state_dict()[name]) for name in before)
    weights, _ = aggregate_client_results(server, [result])
    assert weights == [1.0]
    assert all(torch.equal(result.state_dict[name], server.state_dict()[name]) for name in result.state_dict)
    evaluation = evaluate_model(server, bundle.validation_dataset, torch.device("cpu"), 4, seed=3)
    assert evaluation.examples == 4
    assert server.lif1.membrane is None and server.lif2.membrane is None


def test_official_test_access_is_rejected_before_selection(tmp_path):
    bundle = SyntheticFederatedBundle(_config(tmp_path / "runs"))
    with pytest.raises(OfficialTestAccessError):
        bundle.official_test_dataset(model_selected=False)


def test_interrupted_execution_resumes_with_identical_selections_and_parameters(tmp_path):
    direct_config = _config(tmp_path / "direct")
    direct_bundle = SyntheticFederatedBundle(direct_config)
    direct_run = _new_run(direct_config, direct_bundle)
    direct = train_federated(_model(direct_config), direct_bundle, direct_config, direct_run)
    assert direct["completed"] and direct_bundle.official_test_access_count == 1

    resumed_config = _config(tmp_path / "resumed")
    first_bundle = SyntheticFederatedBundle(resumed_config)
    resumed_run = _new_run(resumed_config, first_bundle)
    partial = train_federated(
        _model(resumed_config),
        first_bundle,
        resumed_config,
        resumed_run,
        stop_after_round=1,
    )
    assert not partial["completed"]
    second_bundle = SyntheticFederatedBundle(resumed_config)
    resumed = train_federated(
        _model(resumed_config),
        second_bundle,
        resumed_config,
        resumed_run,
        resume_checkpoint=resumed_run / "checkpoints" / "last.pt",
    )
    assert resumed["completed"] and second_bundle.official_test_access_count == 1

    direct_rounds = [json.loads(line) for line in (direct_run / "round_metrics.jsonl").read_text().splitlines()]
    resumed_rounds = [json.loads(line) for line in (resumed_run / "round_metrics.jsonl").read_text().splitlines()]
    assert [record["selected_client_ids"] for record in direct_rounds] == [
        record["selected_client_ids"] for record in resumed_rounds
    ]
    direct_state = torch.load(direct_run / "checkpoints" / "last.pt", weights_only=False)["global_model_state"]
    resumed_state = torch.load(resumed_run / "checkpoints" / "last.pt", weights_only=False)["global_model_state"]
    assert all(torch.equal(direct_state[name], resumed_state[name]) for name in direct_state)

    skipped = plan_run(resumed_config, "retry --resume-auto", resume_auto=True)
    assert skipped.skip_completed


def test_incompatible_partition_checkpoint_is_rejected(tmp_path):
    config = _config(tmp_path / "runs")
    bundle = SyntheticFederatedBundle(config)
    run = _new_run(config, bundle)
    train_federated(_model(config), bundle, config, run, stop_after_round=1)
    model = _model(config)
    with pytest.raises(RuntimeError, match="partition_id"):
        load_federated_checkpoint(
            run / "checkpoints" / "last.pt",
            model,
            config,
            run,
            bundle.split_artifact["split_id"],
            "different-partition",
            state_identity(model.state_dict()),
        )
