import copy

import pytest
import torch

from fedapfa.configuration import (
    ConfigurationError,
    load_federated_config,
    load_heterogeneity_context_tasks,
    load_heterogeneity_manifest,
    load_published_fedsnn_manifest,
    validate_federated_config,
)
from fedapfa.models.svgg9_bntt import (
    SVGG9BNTT,
    piecewise_surrogate_spike,
    poisson_rate_encode,
)

HETEROGENEITY_MANIFEST = "experiments/heterogeneity_evaluation/manifest.yaml"
PUBLISHED_MANIFEST = "experiments/published_fedsnn/manifest.yaml"
PUBLISHED_CONFIG = "experiments/published_fedsnn/cifar10/svgg9_bntt_dirichlet_alpha_0_5.yaml"


def test_scientific_manifests_have_exact_task_counts_and_seeds():
    heterogeneity = load_heterogeneity_manifest(HETEROGENEITY_MANIFEST)
    context = load_heterogeneity_context_tasks(HETEROGENEITY_MANIFEST)
    published = load_published_fedsnn_manifest(PUBLISHED_MANIFEST)
    assert len(heterogeneity) == 9
    assert len(context) == 3
    assert len(published) == 3
    assert sorted({value.seed for value in heterogeneity}) == [7, 17, 27]
    assert [value.seed for value in context] == [7, 17, 27]
    assert [value.seed for value in published] == [7, 17, 27]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: value["federated"]["partition"].update(method="unknown"), "unknown partition"),
        (lambda value: value["federated"]["partition"].update(alpha=0.0), "finite and positive"),
        (lambda value: value["federated"].update(clients_per_round=11), "cannot exceed"),
        (lambda value: value["federated"].update(participation_fraction=0.3), "inconsistent"),
        (lambda value: value["federated"].update(optimizer="adam", momentum=None), "requires SGD"),
        (lambda value: value.update(device="cpu"), "require CUDA"),
        (lambda value: value["model"].pop("timesteps"), "timesteps"),
        (lambda value: value["provenance"].update(require_git_commit=False), "provenance"),
        (lambda value: value["subset"].update(train_examples=10), "caps"),
    ],
)
def test_published_configuration_rejections(change, message):
    config = load_federated_config(PUBLISHED_CONFIG)
    changed = copy.deepcopy(config)
    change(changed)
    with pytest.raises((ConfigurationError, KeyError), match=message):
        validate_federated_config(changed)


def test_iid_alpha_and_dataset_model_mismatches_are_rejected():
    iid = load_federated_config(
        "experiments/heterogeneity_evaluation/shd/lif_iid_participation_0_50.yaml"
    )
    iid["federated"]["partition"]["alpha"] = 1.0
    with pytest.raises(ConfigurationError, match="cannot set alpha"):
        validate_federated_config(iid)
    published = load_federated_config(PUBLISHED_CONFIG)
    published["dataset"]["name"] = "shd"
    with pytest.raises(ConfigurationError, match="SHD cannot be paired"):
        validate_federated_config(published)


def _model_config():
    return {
        "dataset": {"channels": 3, "image_size": 8, "classes": 4},
        "model": {
            "channels": [2, 2, 2, 2, 2, 2, 2],
            "average_pool_after_convolution": [2, 4, 7],
            "linear_hidden": 4,
            "timesteps": 3,
            "leak": 0.95,
            "threshold": 1.0,
            "surrogate_scale": 0.3,
            "bntt_momentum": 0.1,
            "bntt_epsilon": 1e-5,
        },
    }


def test_svgg9_bntt_shape_independent_temporal_state_and_finite_gradients():
    model = SVGG9BNTT(_model_config()).train()
    images = torch.rand(2, 3, 8, 8)
    logits, rates = model(images, generator=torch.Generator().manual_seed(7))
    assert logits.shape == (2, 4)
    assert set(rates) == {"conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "linear1"}
    normalization = model.convolution_bntt[0]
    assert normalization.weight.shape == (3, 2)
    assert normalization.running_mean.data_ptr() != normalization.running_mean[1:].data_ptr()
    assert not torch.equal(normalization.running_mean[0], normalization.running_mean[1])
    logits.sum().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    membrane = torch.tensor([0.5, 1.0, 1.5], requires_grad=True)
    piecewise_surrogate_spike(membrane, 1.0, 0.3).sum().backward()
    assert torch.isfinite(membrane.grad).all()


def test_poisson_encoding_is_deterministic_under_fixed_seed():
    images = torch.tensor([[[[0.0, 0.25], [0.75, 1.0]]]])
    first = poisson_rate_encode(images, 6, torch.Generator().manual_seed(19))
    second = poisson_rate_encode(images, 6, torch.Generator().manual_seed(19))
    third = poisson_rate_encode(images, 6, torch.Generator().manual_seed(29))
    assert torch.equal(first, second)
    assert not torch.equal(first, third)
    assert torch.all(first[..., 0, 0] == 0) and torch.all(first[..., 1, 1] == 1)
