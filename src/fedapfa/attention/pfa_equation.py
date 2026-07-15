"""Parameter-free attention faithful to the stated fully-connected equations."""
import torch
from torch import nn
class EquationPFA(nn.Module):
    def __init__(self, lambda_=1e-2, threshold=1.0):
        super().__init__()
        if lambda_ <= 0: raise ValueError("lambda must be positive")
        self.lambda_, self.threshold=lambda_, threshold
    def forward(self, current, return_stats=False):
        mu=current.mean(-1, keepdim=True); variance=(current-mu).square().mean(-1, keepdim=True)
        weight=self.threshold*(current-mu)/((current-mu).square()+2*self.lambda_+2*variance)
        bias=(self.threshold-weight*(current+mu))/2; attention=torch.sigmoid(weight*current+bias); out=current*attention
        return (out, {"mean":mu,"variance":variance,"attention":attention}) if return_stats else out
