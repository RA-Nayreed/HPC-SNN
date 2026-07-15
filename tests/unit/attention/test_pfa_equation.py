import pytest, torch
from fedapfa.attention.pfa_equation import EquationPFA
def test_equation_manual_and_parameter_free():
    value=torch.tensor([[[1.,3.]]]); out=EquationPFA(.01)(value); mu=value.mean(-1,keepdim=True); var=(value-mu).square().mean(-1,keepdim=True); w=(value-mu)/((value-mu).square()+.02+2*var); expected=value*torch.sigmoid(w*value+(1-w*(value+mu))/2); assert torch.allclose(out,expected); assert not list(EquationPFA().parameters())
def test_lambda_and_gradients():
    with pytest.raises(ValueError): EquationPFA(0)
    value=torch.randn(1,2,4,requires_grad=True); EquationPFA()(value).sum().backward(); assert torch.isfinite(value.grad).all()
