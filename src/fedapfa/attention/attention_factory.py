from torch import nn
from .pfa_equation import EquationPFA
from .pfa_public_behavior import PublicBehaviorPFA
def make_attention(variant, lambda_=1e-2):
    if variant == "none": return nn.Identity()
    if variant == "equation": return EquationPFA(lambda_)
    if variant == "public_behavior": return PublicBehaviorPFA(lambda_)
    raise ValueError(f"unknown attention variant: {variant}")
