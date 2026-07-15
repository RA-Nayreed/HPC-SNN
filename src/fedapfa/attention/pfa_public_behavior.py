"""Independent implementation of PfA-SNN commit 0898fc22480c86bccd7f6fccb0d43fdfbd579797."""

import torch
from torch import nn

UPSTREAM_COMMIT = "0898fc22480c86bccd7f6fccb0d43fdfbd579797"


class PublicBehaviorPFA(nn.Module):
    def __init__(self, lambda_: float = 0.01, threshold: float = 1.0):
        super().__init__()
        if lambda_ <= 0:
            raise ValueError("lambda must be positive")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.lambda_ = float(lambda_)
        self.threshold = float(threshold)
        self.last_statistics = None

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        if current.ndim != 3:
            raise ValueError("PfA current must have shape [batch, time, neurons]")
        mean = current.mean(dim=-1, keepdim=True)
        squared_deviation = (current - mean).square()
        weight = self.threshold * (current - mean) / (squared_deviation + 2 * self.lambda_ + 2 * squared_deviation)
        bias = (self.threshold - weight * (current + mean)) / 2
        attention = torch.sigmoid(weight * current + bias)
        self.last_statistics = {
            "current_mean": float(mean.detach().mean()),
            "attention_mean": float(attention.detach().mean()),
        }
        return current * attention
