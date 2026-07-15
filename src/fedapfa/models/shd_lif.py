import torch
from torch import nn
from fedapfa.attention import make_attention
from fedapfa.neurons.lif import LIF
class AudioLIFSNN(nn.Module):
    def __init__(self, features, hidden, classes, attention="none", lambda_=1e-2, dropout=.4):
        super().__init__(); a,b=hidden; self.linear1=nn.Linear(features,a); self.linear2=nn.Linear(a,b); self.readout=nn.Linear(b,classes); self.attention1=make_attention(attention,lambda_); self.attention2=make_attention(attention,lambda_); self.lif1=LIF(); self.lif2=LIF(); self.dropout=nn.Dropout(dropout)
    def forward(self, inputs, lengths):
        valid=torch.arange(inputs.shape[1],device=inputs.device)[None,:]<lengths[:,None]
        h1=self.lif1(self.attention1(self.linear1(inputs)),valid); h2=self.lif2(self.attention2(self.linear2(self.dropout(h1))),valid)
        return (self.readout(h2)*valid.unsqueeze(-1)).sum(1), {"lif1":h1.sum()/valid.sum().clamp_min(1),"lif2":h2.sum()/valid.sum().clamp_min(1)}
