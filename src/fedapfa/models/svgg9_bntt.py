"""Independent S-VGG9 with Batch Normalization Through Time."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _PiecewiseSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane: torch.Tensor, threshold: float, scale: float) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        ctx.scale = scale
        return (membrane > threshold).to(membrane.dtype)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        (membrane,) = ctx.saved_tensors
        distance = (membrane - ctx.threshold).abs() / ctx.threshold
        surrogate = ctx.scale * torch.clamp(1.0 - distance, min=0.0)
        return gradient * surrogate, None, None


def piecewise_surrogate_spike(membrane: torch.Tensor, threshold: float, scale: float) -> torch.Tensor:
    """Apply the paper's triangular piecewise surrogate derivative."""

    return _PiecewiseSurrogate.apply(membrane, float(threshold), float(scale))


def signed_poisson_encode(
    images: torch.Tensor,
    timesteps: int,
    generator: torch.Generator,
    rescale_factor: float,
) -> torch.Tensor:
    """Encode signed image intensities with the released Fed-SNN rule."""

    if generator is None:
        raise ValueError("signed Poisson encoding requires an explicit generator")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    if images.ndim != 4 or not images.is_floating_point():
        raise ValueError("images must have floating shape [batch, channels, height, width]")
    if not isinstance(timesteps, int) or timesteps <= 0:
        raise ValueError("timesteps must be a positive integer")
    if not isinstance(rescale_factor, (int, float)) or not torch.isfinite(torch.tensor(float(rescale_factor))):
        raise ValueError("Poisson rescale factor must be finite")
    if float(rescale_factor) <= 0:
        raise ValueError("Poisson rescale factor must be positive")
    if not bool(torch.isfinite(images).all()):
        raise ValueError("signed Poisson inputs must be finite")
    if bool(torch.any(images < -1)) or bool(torch.any(images > 1)):
        raise ValueError("signed Poisson inputs must be in [-1, 1]")
    random_values = torch.rand(
        (timesteps, *images.shape),
        dtype=images.dtype,
        device=images.device,
        generator=generator,
    )
    spikes = (random_values * float(rescale_factor) <= images.abs().unsqueeze(0)).to(images.dtype)
    return spikes * images.sign().unsqueeze(0)


def poisson_rate_encode(
    images: torch.Tensor,
    timesteps: int,
    generator: torch.Generator,
    rescale_factor: float = 2.0,
) -> torch.Tensor:
    """Compatibility name for the signed Fed-SNN Poisson encoder."""

    return signed_poisson_encode(images, timesteps, generator, rescale_factor)


class TemporalBatchNorm(nn.Module):
    """Batch normalization with independent parameters and statistics per timestep."""

    def __init__(self, timesteps: int, features: int, momentum: float, epsilon: float):
        super().__init__()
        self.timesteps = timesteps
        self.features = features
        self.momentum = momentum
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(timesteps, features))
        self.register_buffer("running_mean", torch.zeros(timesteps, features))
        self.register_buffer("running_variance", torch.ones(timesteps, features))

    def forward(self, inputs: torch.Tensor, timestep: int) -> torch.Tensor:
        if not 0 <= timestep < self.timesteps:
            raise IndexError("BNTT timestep is outside the configured range")
        elements_per_feature = inputs.numel() // inputs.shape[1]
        use_batch_statistics = self.training and elements_per_feature > 1
        return F.batch_norm(
            inputs,
            self.running_mean[timestep],
            self.running_variance[timestep],
            self.weight[timestep],
            None,
            use_batch_statistics,
            self.momentum,
            self.epsilon,
        )


class SVGG9BNTT(nn.Module):
    """Seven convolutional and two linear layers with temporal batch normalization."""

    def __init__(self, config: dict):
        super().__init__()
        dataset = config["dataset"]
        model = config["model"]
        channels = list(model["channels"])
        self.timesteps = int(model["timesteps"])
        self.leak = float(model["leak"])
        self.threshold = float(model["threshold"])
        self.surrogate_scale = float(model["surrogate_scale"])
        self.input_encoding = model["input_encoding"]
        self.poisson_rescale_factor = float(model["poisson_rescale_factor"])
        self.readout_rule = model["readout"]
        self.weight_initialization = model["weight_initialization"]
        if self.input_encoding != "signed_poisson":
            raise ValueError(f"unsupported S-VGG9 input encoding: {self.input_encoding}")
        if self.readout_rule != "temporal_mean":
            raise ValueError(f"unsupported S-VGG9 readout: {self.readout_rule}")
        if self.weight_initialization != "xavier_uniform_gain_2":
            raise ValueError(f"unsupported S-VGG9 weight initialization: {self.weight_initialization}")
        self.pool_after = {int(value) - 1 for value in model["average_pool_after_convolution"]}
        momentum = float(model["bntt_momentum"])
        epsilon = float(model["bntt_epsilon"])

        convolution_layers = []
        normalization_layers = []
        input_channels = int(dataset["channels"])
        spatial = int(dataset["image_size"])
        self.input_shape = (input_channels, spatial, spatial)
        for index, output_channels in enumerate(channels):
            convolution_layers.append(
                nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
            )
            normalization_layers.append(TemporalBatchNorm(self.timesteps, output_channels, momentum, epsilon))
            input_channels = output_channels
            if index in self.pool_after:
                spatial //= 2
        self.convolutions = nn.ModuleList(convolution_layers)
        self.convolution_bntt = nn.ModuleList(normalization_layers)
        self.average_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(channels[-1] * spatial * spatial, int(model["linear_hidden"]), bias=False)
        self.linear1_bntt = TemporalBatchNorm(self.timesteps, int(model["linear_hidden"]), momentum, epsilon)
        self.readout = nn.Linear(int(model["linear_hidden"]), int(dataset["classes"]), bias=False)
        self._initialize_weights()
        self.model_metadata = {
            "class": type(self).__name__,
            "convolution_channels": channels,
            "average_pool_after_convolution": sorted(index + 1 for index in self.pool_after),
            "linear_hidden": int(model["linear_hidden"]),
            "classes": int(dataset["classes"]),
            "timesteps": self.timesteps,
            "input_encoding": self.input_encoding,
            "poisson_rescale_factor": self.poisson_rescale_factor,
            "readout": self.readout_rule,
            "weight_initialization": self.weight_initialization,
        }

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight, gain=2)

    def reset_state(self) -> None:
        """State is local to each forward call, so no persistent membrane remains."""

    def _lif(self, current: torch.Tensor, membrane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        membrane = self.leak * membrane + current
        spikes = piecewise_surrogate_spike(membrane, self.threshold, self.surrogate_scale)
        membrane = membrane - spikes.detach() * self.threshold
        return spikes, membrane

    def forward(
        self,
        images: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if images.ndim != 4 or tuple(images.shape[1:]) != self.input_shape:
            raise ValueError(f"S-VGG9 BNTT inputs must have shape [batch, {self.input_shape}]")
        encoded = signed_poisson_encode(
            images,
            self.timesteps,
            generator,
            self.poisson_rescale_factor,
        )
        convolution_membranes: list[torch.Tensor | None] = [None] * len(self.convolutions)
        linear_membrane: torch.Tensor | None = None
        output_membrane: torch.Tensor | None = None
        rate_sums = {
            **{f"conv{index + 1}": images.new_zeros(()) for index in range(len(self.convolutions))},
            "linear1": images.new_zeros(()),
        }
        for timestep in range(self.timesteps):
            spikes = encoded[timestep]
            for index, (convolution, normalization) in enumerate(
                zip(self.convolutions, self.convolution_bntt, strict=True)
            ):
                current = normalization(convolution(spikes), timestep)
                if convolution_membranes[index] is None:
                    convolution_membranes[index] = torch.zeros_like(current)
                spikes, convolution_membranes[index] = self._lif(current, convolution_membranes[index])
                rate_sums[f"conv{index + 1}"] = rate_sums[f"conv{index + 1}"] + spikes.mean()
                if index in self.pool_after:
                    spikes = self.average_pool(spikes)
            current = self.linear1_bntt(self.linear1(torch.flatten(spikes, 1)), timestep)
            if linear_membrane is None:
                linear_membrane = torch.zeros_like(current)
            spikes, linear_membrane = self._lif(current, linear_membrane)
            rate_sums["linear1"] = rate_sums["linear1"] + spikes.mean()
            current_output = self.readout(spikes)
            output_membrane = current_output if output_membrane is None else output_membrane + current_output
        if output_membrane is None:
            raise RuntimeError("S-VGG9 BNTT produced no output timesteps")
        rates = {name: value / self.timesteps for name, value in rate_sums.items()}
        return output_membrane / self.timesteps, rates
