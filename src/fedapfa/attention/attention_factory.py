from torch import nn

from .pfa_equation import EquationPFA
from .pfa_public_behavior import PublicBehaviorPFA


def make_attention(variant: str, lambda_: float = 0.01, threshold: float = 1.0) -> nn.Module:
    if variant == "none":
        return nn.Identity()
    if variant == "equation":
        return EquationPFA(lambda_, threshold)
    if variant == "public_behavior":
        return PublicBehaviorPFA(lambda_, threshold)
    raise ValueError(f"unknown attention variant: {variant}")
