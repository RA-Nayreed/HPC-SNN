"""SpikingJelly-compatible ATan surrogate with binary forward output."""

import math

import torch


class _ATanSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(input_)
        ctx.alpha = alpha
        return (input_ >= 0).to(input_.dtype)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        (input_,) = ctx.saved_tensors
        denominator = 1 + (math.pi * ctx.alpha * input_ / 2).square()
        return gradient * ctx.alpha / (2 * denominator), None


def atan_spike(input_: torch.Tensor, alpha: float = 5.0) -> torch.Tensor:
    if alpha <= 0:
        raise ValueError("surrogate alpha must be positive")
    return _ATanSpike.apply(input_, float(alpha))
