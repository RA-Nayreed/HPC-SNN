"""Equation-faithful Parameter-free Attention for [batch, time, neurons]."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PFAStatistics:
    current_mean: float
    variance_mean: float
    attention_mean: float


class EquationPFA(nn.Module):
    def __init__(self, lambda_: float = 0.01, threshold: float = 1.0):
        super().__init__()
        if lambda_ <= 0:
            raise ValueError("lambda must be positive")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.lambda_ = float(lambda_)
        self.threshold = float(threshold)
        self.last_statistics = None

    def forward(self, current: torch.Tensor, return_stats: bool = False):
        if current.ndim != 3:
            raise ValueError("PfA current must have shape [batch, time, neurons]")
        mean = current.mean(dim=-1, keepdim=True)
        deviation = current - mean
        variance = deviation.square().mean(dim=-1, keepdim=True)
        weight = self.threshold * deviation / (deviation.square() + 2 * self.lambda_ + 2 * variance)
        bias = (self.threshold - weight * (current + mean)) / 2
        attention = torch.sigmoid(weight * current + bias)
        output = current * attention
        self.last_statistics = {
            "current_mean": float(mean.detach().mean()),
            "variance_mean": float(variance.detach().mean()),
            "attention_mean": float(attention.detach().mean()),
        }
        if not return_stats:
            return output
        stats = PFAStatistics(
            float(mean.detach().mean()), float(variance.detach().mean()), float(attention.detach().mean())
        )
        return output, stats
