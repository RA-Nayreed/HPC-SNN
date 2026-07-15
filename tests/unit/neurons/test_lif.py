import torch
from fedapfa.neurons.lif import LIF
def test_manual_lif_and_reset():
    lif=LIF(tau_ms=20,dt_ms=10,threshold=1); current=torch.tensor([[[1.],[1.]]]); spikes=lif(current)
    assert torch.allclose(spikes,torch.tensor([[[1.],[0.]]]))
def test_surrogate_gradient_finite():
    current=torch.tensor([[[1.]]],requires_grad=True); LIF()(current).sum().backward(); assert torch.isfinite(current.grad).all()
