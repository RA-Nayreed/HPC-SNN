import pytest
import torch

from fedapfa.configuration import load_config
from fedapfa.models.model_factory import make_model
from fedapfa.training.centralized import DeviceUnavailableError, make_optimizer, resolve_device
from fedapfa.training.checkpointing import load_checkpoint, save_checkpoint


def test_cuda_request_fails_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(DeviceUnavailableError, match="requests CUDA"):
        resolve_device("cuda")


def test_checkpoint_saves_and_restores_complete_state(tmp_path):
    config = load_config("tests/data/configurations/centralized/shd_memorization_validation.yaml")
    model = make_model(config)
    optimizer = make_optimizer(model, config)
    path = tmp_path / "state.pt"
    original = next(model.parameters()).detach().clone()
    save_checkpoint(path, model, optimizer, None, config, 2, 11, 0.75)
    with torch.no_grad():
        next(model.parameters()).add_(1)
    state = load_checkpoint(path, model, optimizer)
    assert (
        state["epoch"] == 2
        and state["global_step"] == 11
        and state["best_selection_accuracy"] == 0.75
        and state["model_class"] == type(model).__name__
        and torch.allclose(next(model.parameters()), original)
    )


def test_dcls_optimizer_uses_delay_multiplier_when_available():
    config = load_config("experiments/centralized/shd/dcls_published_protocol.yaml")
    try:
        model = make_model(config)
    except Exception:
        pytest.skip("dcls==0.1.1 unavailable or incompatible")
    optimizer = make_optimizer(model, config)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"] * 100)
    model.eval()
    logits, _ = model(torch.randn(2, 12, 140), torch.tensor([12, 8]))
    logits.sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.delay_parameters()
    )
