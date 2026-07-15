import torch
from fedapfa.attention import EquationPFA, PublicBehaviorPFA
def test_public_variant_shape_and_difference():
    value=torch.tensor([[[0.,1.,10.]]]); assert EquationPFA()(value).shape==value.shape; assert not torch.allclose(EquationPFA()(value),PublicBehaviorPFA()(value))
