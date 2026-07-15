import torch

from fedapfa.configuration import load_config
from fedapfa.models.model_factory import make_model


def test_real_model_forward_backward():
    config = load_config("tests/data/configurations/centralized/shd_memorization_validation.yaml")
    model = make_model(config)
    inputs = torch.rand(2, 3, 140)
    lengths = torch.tensor([3, 2])
    logits, rates = model(inputs, lengths)
    logits.sum().backward()
    assert (
        logits.shape == (2, 20)
        and all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        and all(0 <= float(rate.detach()) <= 1 for rate in rates.values())
    )
