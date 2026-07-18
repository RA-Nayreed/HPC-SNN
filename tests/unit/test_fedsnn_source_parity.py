import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

import fedapfa.models.svgg9_bntt as svgg_module
from fedapfa.configuration import load_federated_config, load_resolved_config
from fedapfa.datasets.centralized_split import stratified_split
from fedapfa.datasets.cifar10 import normalize_cifar10_tensor
from fedapfa.datasets.fedsnn_partition import (
    fedsnn_balanced_label_dirichlet_partition,
    fedsnn_random_iid_partition,
)
from fedapfa.federated.checkpointing import (
    configuration_identity,
    load_federated_checkpoint,
)
from fedapfa.federated.client import _loader, evaluate_model
from fedapfa.federated.randomness import resolved_seeds
from fedapfa.models.svgg9_bntt import SVGG9BNTT, TemporalBatchNorm, signed_poisson_encode
from fedapfa.training.optimization import learning_rate_for_round


def _small_model_config(timesteps=3):
    return {
        "dataset": {"channels": 3, "image_size": 8, "classes": 4},
        "model": {
            "name": "svgg9_bntt",
            "channels": [2, 2, 2, 2, 2, 2, 2],
            "average_pool_after_convolution": [2, 4, 7],
            "linear_hidden": 6,
            "timesteps": timesteps,
            "leak": 0.95,
            "threshold": 1.0,
            "surrogate_scale": 0.3,
            "bntt_momentum": 0.1,
            "bntt_epsilon": 0.0001,
            "input_encoding": "signed_poisson",
            "poisson_rescale_factor": 2.0,
            "readout": "temporal_mean",
            "weight_initialization": "xavier_uniform_gain_2",
        },
    }


def test_signed_normalization_matches_channelwise_half_mean_and_standard_deviation():
    values = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    normalized = normalize_cifar10_tensor(values, "signed_minus_one_one")
    assert torch.equal(normalized, torch.tensor([-1.0, 0.0, 1.0]))
    assert torch.equal(normalized, values.mul(2.0).sub(1.0))
    assert torch.isfinite(normalized).all()


def test_signed_normalization_rejects_unknown_nonfinite_and_out_of_range_values():
    with pytest.raises(ValueError, match="unsupported"):
        normalize_cifar10_tensor(torch.zeros(1), "unknown")
    with pytest.raises(ValueError, match="NaN or infinity"):
        normalize_cifar10_tensor(torch.tensor([float("nan")]), "signed_minus_one_one")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        normalize_cifar10_tensor(torch.tensor([1.1]), "signed_minus_one_one")


def test_signed_poisson_exact_rule_sign_values_and_reproducibility():
    images = torch.tensor([[[[0.0, 0.4], [-0.7, 1.0]]]], dtype=torch.float32)
    expected_generator = torch.Generator().manual_seed(119)
    random_values = torch.rand((5, *images.shape), generator=expected_generator)
    expected = (random_values * 2.0 <= images.abs().unsqueeze(0)).float()
    expected = expected * images.sign().unsqueeze(0)
    actual = signed_poisson_encode(images, 5, torch.Generator().manual_seed(119), rescale_factor=2.0)
    repeated = signed_poisson_encode(images, 5, torch.Generator().manual_seed(119), rescale_factor=2.0)
    assert torch.equal(actual, expected)
    assert torch.equal(actual, repeated)
    assert set(actual.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert torch.all(actual[..., 0, 0] == 0)
    assert set(actual[..., 0, 1].unique().tolist()) <= {0.0, 1.0}
    assert set(actual[..., 1, 0].unique().tolist()) <= {-1.0, 0.0}


@pytest.mark.parametrize(
    "images,message",
    [
        (torch.tensor([[[[float("inf")]]]]), "finite"),
        (torch.tensor([[[[1.01]]]]), r"\[-1, 1\]"),
        (torch.ones(2, 3), "shape"),
        (torch.ones(1, 1, 1, 1, dtype=torch.int64), "floating"),
    ],
)
def test_signed_poisson_rejects_invalid_inputs(images, message):
    with pytest.raises(ValueError, match=message):
        signed_poisson_encode(images, 2, torch.Generator().manual_seed(1), 2.0)
    with pytest.raises(ValueError, match="explicit generator"):
        signed_poisson_encode(torch.zeros(1, 1, 1, 1), 2, None, 2.0)


def test_temporal_mean_readout_is_inside_the_gradient_path(monkeypatch):
    model = SVGG9BNTT(_small_model_config(timesteps=3)).eval()
    encoded = torch.ones(3, 2, 3, 8, 8)
    monkeypatch.setattr(svgg_module, "signed_poisson_encode", lambda *args: encoded)
    outputs = []
    original_forward = model.readout.forward

    def capture(inputs):
        value = original_forward(inputs)
        outputs.append(value)
        return value

    monkeypatch.setattr(model.readout, "forward", capture)
    logits, _ = model(torch.zeros(2, 3, 8, 8), generator=torch.Generator().manual_seed(2))
    assert torch.allclose(logits, torch.stack(outputs).sum(0) / 3)
    gradients = torch.autograd.grad(logits.sum(), outputs, retain_graph=True)
    for gradient in gradients:
        assert torch.equal(gradient, torch.full_like(gradient, 1 / 3))


def test_xavier_gain_two_is_applied_once_to_every_conv_and_linear(monkeypatch):
    calls = []
    original = nn.init.xavier_uniform_

    def record(tensor, gain=1.0, generator=None):
        calls.append((tensor, gain))
        return original(tensor, gain=gain, generator=generator)

    monkeypatch.setattr(nn.init, "xavier_uniform_", record)
    model = SVGG9BNTT(_small_model_config())
    affine_layers = [module for module in model.modules() if isinstance(module, (nn.Conv2d, nn.Linear))]
    assert len(affine_layers) == 9
    assert len(calls) == 9 and all(gain == 2 for _, gain in calls)
    assert all(layer.bias is None for layer in affine_layers)
    assert all(torch.equal(module.weight, torch.ones_like(module.weight)) for module in model.convolution_bntt)
    assert torch.equal(model.linear1_bntt.weight, torch.ones_like(model.linear1_bntt.weight))


def test_model_construction_is_deterministic_for_the_same_resolved_seed():
    torch.manual_seed(71)
    first = SVGG9BNTT(_small_model_config()).state_dict()
    torch.manual_seed(71)
    second = SVGG9BNTT(_small_model_config()).state_dict()
    assert all(torch.equal(first[name], second[name]) for name in first)


@pytest.mark.parametrize("shape", [(5, 3, 4, 4), (5, 3)])
def test_temporal_batch_norm_matches_equivalent_pytorch_training_and_evaluation(shape):
    torch.manual_seed(14)
    inputs = torch.randn(*shape)
    temporal = TemporalBatchNorm(2, 3, momentum=0.1, epsilon=0.0001)
    reference_class = nn.BatchNorm2d if len(shape) == 4 else nn.BatchNorm1d
    reference = reference_class(3, eps=0.0001, momentum=0.1, affine=True)
    reference.bias = None
    reference.weight.data.copy_(temporal.weight[0])
    actual = temporal(inputs, 0)
    expected = reference(inputs)
    assert torch.equal(actual, expected)
    assert torch.equal(temporal.running_mean[0], reference.running_mean)
    assert torch.equal(temporal.running_variance[0], reference.running_var)
    before_other_scale = temporal.weight[1].detach().clone()
    temporal.weight.data[0].add_(2)
    assert torch.equal(temporal.weight[1], before_other_scale)
    assert temporal.epsilon == 0.0001
    assert set(dict(temporal.named_parameters())) == {"weight"}
    temporal.eval()
    reference.eval()
    reference.weight.data.copy_(temporal.weight[0])
    assert torch.equal(temporal(inputs, 0), reference(inputs))


class _ImageDataset(Dataset):
    fedapfa_batch_kind = "image"

    def __init__(self, count=5):
        self.images = torch.linspace(-1, 1, count * 3 * 2 * 2).reshape(count, 3, 2, 2)
        self.labels = torch.arange(count) % 2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class _ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(12, 2)

    def forward(self, images, *, generator):
        torch.rand((), generator=generator)
        return self.linear(images.flatten(1)), {"image": images.abs().mean()}


def test_local_loader_drops_remainders_but_evaluation_keeps_every_example():
    dataset = _ImageDataset(5)
    local = _loader(dataset, 2, True, 9, 0, False, drop_last=True)
    retained = _loader(dataset, 2, False, 9, 0, False, drop_last=False)
    assert sum(len(labels) for _, labels in local) == 4
    assert sum(len(labels) for _, labels in retained) == 5
    validation = evaluate_model(_ImageModel(), dataset, torch.device("cpu"), 2, 7)
    official_test = evaluate_model(_ImageModel(), dataset, torch.device("cpu"), 2, 8)
    assert validation.examples == official_test.examples == 5


def test_client_loader_preserves_explicit_pipeline_controls():
    loader = _loader(
        _ImageDataset(5),
        2,
        True,
        9,
        1,
        True,
        pin_memory=True,
        prefetch_factor=3,
    )
    assert loader.num_workers == 1
    assert loader.persistent_workers
    assert loader.pin_memory
    assert loader.prefetch_factor == 3


def test_source_partitions_are_deterministic_complete_nonoverlapping_and_named():
    labels = np.tile(np.arange(10, dtype=np.int64), 100)
    eligible = np.arange(len(labels), dtype=np.int64)
    common = {
        "labels": labels,
        "eligible_indices": eligible,
        "clients": 10,
        "minimum_size": 10,
        "seed": 31,
        "validation_split_id": "split",
        "dataset_identity": {"name": "synthetic"},
    }
    first = fedsnn_balanced_label_dirichlet_partition(**common, alpha=0.5, maximum_attempts=1000)
    second = fedsnn_balanced_label_dirichlet_partition(**common, alpha=0.5, maximum_attempts=1000)
    iid = fedsnn_random_iid_partition(**common)
    assert first.partition_id == second.partition_id
    assert first.artifact["method"] == "fedsnn_balanced_label_dirichlet"
    assert iid.artifact["method"] == "fedsnn_random_iid"
    for partition in (first, iid):
        assigned = [index for values in partition.client_indices.values() for index in values]
        assert sorted(assigned) == eligible.tolist()
        assert len(assigned) == len(set(assigned))
        assert min(len(values) for values in partition.client_indices.values()) >= 10
        assert all("class_counts" in client for client in partition.artifact["clients"])
        assert partition.artifact["numpy_random_state_seed"] == common["seed"] % (2**32)


def test_zero_validation_exposes_all_50000_examples_to_table_i_partitions():
    labels = np.repeat(np.arange(10, dtype=np.int64), 5000)
    training, validation = stratified_split(labels, validation_fraction=0.0, seed=7)
    assert training.tolist() == list(range(50000))
    assert validation.size == 0
    common = {
        "labels": labels,
        "eligible_indices": training,
        "clients": 10,
        "minimum_size": 10,
        "seed": resolved_seeds(
            load_federated_config("experiments/published_fedsnn/cifar10/paper_reported_iid_evaluation.yaml")
        )["partition"],
        "validation_split_id": "all-training",
        "dataset_identity": {"name": "cifar10-fixture"},
    }
    iid = fedsnn_random_iid_partition(**common)
    noniid = fedsnn_balanced_label_dirichlet_partition(**common, alpha=0.5, maximum_attempts=1000)
    for partition in (iid, noniid):
        assigned = [index for indices in partition.client_indices.values() for index in indices]
        assert len(assigned) == 50000
        assert sorted(assigned) == training.tolist()
        assert len(set(assigned)) == 50000
        assert all(len(client["class_counts"]) == 10 for client in partition.artifact["clients"])
        assert partition.artifact["numpy_random_state_seed"] == common["seed"] % (2**32)
    assert {len(indices) for indices in iid.client_indices.values()} == {5000}


def test_learning_rate_changes_only_after_each_configured_boundary():
    federation = {
        "learning_rate": 0.1,
        "learning_rate_reduction_rounds": [40, 60, 80],
        "learning_rate_reduction_factor": 5,
    }
    ranges = ((1, 40, 0.1), (41, 60, 0.02), (61, 80, 0.004), (81, 100, 0.0008))
    for first_round, last_round, learning_rate in ranges:
        for round_number in range(first_round, last_round + 1):
            assert learning_rate_for_round(federation, round_number) == pytest.approx(learning_rate)


def test_prior_configuration_checkpoint_identity_is_rejected(tmp_path):
    corrected = load_federated_config("experiments/published_fedsnn/cifar10/paper_reported_iid_evaluation.yaml")
    prior = load_resolved_config(
        "experiments/published_fedsnn/history/cifar10_svgg9_bntt_independent_implementation.yaml"
    )
    model = nn.Linear(2, 2)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "git.json").write_text(json.dumps({"commit": "test-commit"}), encoding="utf-8")
    checkpoint = tmp_path / "prior.pt"
    torch.save(
        {
            "model_class": type(model).__name__,
            "configuration_id": configuration_identity(prior),
            "resolved_config": prior,
            "aggregation_weighting": "example_count",
            "checkpoint_selection": "best_validation",
            "git_commit": "test-commit",
            "split_id": "split",
            "partition_id": "partition",
            "model_initialization_id": "initialization",
            "next_round": 2,
            "global_model_state": model.state_dict(),
        },
        checkpoint,
    )
    with pytest.raises(RuntimeError, match="configuration_id is incompatible"):
        load_federated_checkpoint(
            checkpoint,
            model,
            corrected,
            run_dir,
            "split",
            "partition",
            "initialization",
        )
    assert corrected["output_root"] != prior["output_root"]


def test_one_step_has_finite_nonzero_gradients_and_configured_sgd_update(monkeypatch):
    torch.manual_seed(81)
    model = SVGG9BNTT(_small_model_config(timesteps=2)).train()
    encoded = torch.randint(0, 2, (2, 8, 3, 8, 8), dtype=torch.float32).mul(2).sub(1)
    monkeypatch.setattr(svgg_module, "signed_poisson_encode", lambda *args: encoded)
    labels = torch.arange(8) % 4
    logits, _ = model(torch.zeros(8, 3, 8, 8), generator=torch.Generator().manual_seed(5))
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    gradients = (
        model.convolutions[0].weight.grad,
        model.convolution_bntt[0].weight.grad,
        model.readout.weight.grad,
    )
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in gradients)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.95)
    before = model.convolutions[0].weight.detach().clone()
    gradient = model.convolutions[0].weight.grad.detach().clone()
    optimizer.step()
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert optimizer.param_groups[0]["momentum"] == 0.95
    assert torch.allclose(model.convolutions[0].weight, before - 0.1 * gradient)
