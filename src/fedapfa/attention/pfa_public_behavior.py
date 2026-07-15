"""Separate observable public-example PfA behavior implementation."""
import torch
from torch import nn
class PublicBehaviorPFA(nn.Module):
    def __init__(self, lambda_=1e-2, threshold=1.0):
        super().__init__()
        if lambda_ <= 0: raise ValueError("lambda must be positive")
        self.lambda_, self.threshold=lambda_, threshold
    def forward(self, current):
        mu=current.mean(-1, keepdim=True)
        variance=current.var(-1, keepdim=True, unbiased=True)
        weight=self.threshold*(current-mu)/((current-mu).square()+2*self.lambda_+2*variance)
        bias=(self.threshold-weight*(current+mu))/2
        return current*torch.sigmoid(weight*current+bias)
