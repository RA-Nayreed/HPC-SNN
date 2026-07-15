"""ATan surrogate gradient with binary forward spikes."""
import torch

class _ATanSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(input_); ctx.alpha = alpha
        return (input_ >= 0).to(input_.dtype)
    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (input_,) = ctx.saved_tensors
        return grad * (ctx.alpha / 2) / (1 + (ctx.alpha * input_).square()), None

def atan_spike(input_: torch.Tensor, alpha: float = 5.0) -> torch.Tensor:
    if alpha <= 0: raise ValueError("surrogate alpha must be positive")
    return _ATanSpike.apply(input_, alpha)
