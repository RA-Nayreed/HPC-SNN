import copy

import pytest
import torch

from fedapfa.configuration import load_config
from fedapfa.models.model_factory import make_model
from fedapfa.models.shd_dcls import DCLSSHDSNN, DCLSUnavailableError, dcls_available
from fedapfa.models.shd_lif import AudioLIFSNN


def test_factory_dimensions_metadata_and_parameter_free_attention():
    config = load_config("experiments/week01_pfa_reproduction/01_tiny_overfit.yaml")
    plain = make_model(config)
    assert isinstance(plain, AudioLIFSNN) and plain.model_metadata["hidden_dims"] == [256, 256]
    attended = copy.deepcopy(config)
    attended["model"]["attention"]["variant"] = "equation"
    assert sum(p.numel() for p in plain.parameters()) == sum(p.numel() for p in make_model(attended).parameters())


def test_shd_and_ssc_shapes_and_padded_logits():
    for path, classes in [
        ("experiments/week01_pfa_reproduction/01_tiny_overfit.yaml", 20),
        ("experiments/week01_pfa_reproduction/08_ssc_tiny_overfit.yaml", 35),
    ]:
        model = make_model(load_config(path)).eval()
        inputs = torch.randn(2, 3, 140)
        lengths = torch.tensor([3, 1])
        logits, rates = model(inputs, lengths)
        assert logits.shape == (2, classes) and set(rates) == {"layer1", "layer2"}
        changed = inputs.clone()
        changed[1, 1:] = 1000
        assert torch.allclose(model(inputs, lengths)[0][1], model(changed, lengths)[0][1])


def test_state_does_not_leak():
    model = make_model(load_config("experiments/week01_pfa_reproduction/01_tiny_overfit.yaml")).eval()
    inputs = torch.randn(1, 2, 140)
    lengths = torch.tensor([2])
    assert torch.allclose(model(inputs, lengths)[0], model(inputs, lengths)[0])


def test_dcls_has_no_fallback():
    config = load_config("experiments/week01_pfa_reproduction/03_dcls_shd.yaml")
    if dcls_available():
        assert isinstance(make_model(config), DCLSSHDSNN)
    else:
        with pytest.raises(DCLSUnavailableError, match="no ordinary-LIF fallback"):
            make_model(config)
