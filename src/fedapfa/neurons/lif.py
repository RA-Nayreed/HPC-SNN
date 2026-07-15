"""u[t]=(1-dt/tau)u[t-1]+I[t], binary spike, detached subtractive reset."""
import torch
from torch import nn
from .surrogate import atan_spike
class LIF(nn.Module):
    def __init__(self, tau_ms=10.05, dt_ms=10.0, threshold=1.0, alpha=5.0):
        super().__init__()
        if tau_ms <= dt_ms or threshold <= 0: raise ValueError("invalid LIF parameters")
        self.decay, self.threshold, self.alpha = 1-dt_ms/tau_ms, threshold, alpha
        self.membrane = None
    def reset_state(self): self.membrane = None
    def forward(self, current, valid=None):
        self.reset_state(); membrane = torch.zeros_like(current[:, 0]); output=[]
        for t in range(current.shape[1]):
            membrane = self.decay * membrane + current[:, t]
            spike = atan_spike(membrane-self.threshold, self.alpha)
            membrane = membrane - spike.detach()*self.threshold
            if valid is not None:
                gate=valid[:, t].unsqueeze(-1); spike=spike*gate; membrane=membrane*gate
            output.append(spike)
        self.membrane=membrane.detach()
        return torch.stack(output, 1)
