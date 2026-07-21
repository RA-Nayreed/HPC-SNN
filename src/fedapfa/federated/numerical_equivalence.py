"""Exact and tolerance-aware model-state comparison."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class NumericalEquivalence:
    structural_identity: bool
    mathematical_equivalence: bool
    bitwise_parameter_identity: bool
    maximum_absolute_parameter_difference: float
    maximum_relative_parameter_difference: float
    finite_parameters: bool
    absolute_tolerance: float
    relative_tolerance: float

    def record(self) -> dict:
        return asdict(self)


def classify_model_states(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> NumericalEquivalence:
    """Classify structure, bits, finiteness, and elementwise numeric bounds."""

    if any(
        isinstance(value, bool) or not math.isfinite(value) or value < 0
        for value in (absolute_tolerance, relative_tolerance)
    ):
        raise ValueError("numeric tolerances must be finite and nonnegative")
    structural = list(reference) == list(candidate) and all(
        reference[name].shape == candidate[name].shape and reference[name].dtype == candidate[name].dtype
        for name in reference
        if name in candidate
    )
    if not structural:
        return NumericalEquivalence(
            structural_identity=False,
            mathematical_equivalence=False,
            bitwise_parameter_identity=False,
            maximum_absolute_parameter_difference=math.inf,
            maximum_relative_parameter_difference=math.inf,
            finite_parameters=False,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )

    bitwise = True
    finite = True
    maximum_absolute = 0.0
    maximum_relative = 0.0
    mathematical = True
    for name, reference_value in reference.items():
        candidate_value = candidate[name]
        bitwise = bitwise and torch.equal(reference_value, candidate_value)
        if reference_value.is_floating_point() or reference_value.is_complex():
            reference_high = (
                reference_value.detach()
                .cpu()
                .to(dtype=torch.complex128 if reference_value.is_complex() else torch.float64)
            )
            candidate_high = candidate_value.detach().cpu().to(dtype=reference_high.dtype)
            values_finite = bool(torch.isfinite(reference_high).all() and torch.isfinite(candidate_high).all())
            finite = finite and values_finite
            if not values_finite:
                mathematical = False
                maximum_absolute = maximum_relative = math.inf
                continue
            absolute = (candidate_high - reference_high).abs()
            denominator = reference_high.abs()
            relative = torch.where(
                denominator > 0,
                absolute / denominator,
                torch.where(absolute == 0, torch.zeros_like(absolute), torch.full_like(absolute, math.inf)),
            )
            if absolute.numel():
                maximum_absolute = max(maximum_absolute, float(absolute.max()))
                maximum_relative = max(maximum_relative, float(relative.max()))
            mathematical = mathematical and bool(
                torch.allclose(
                    reference_high,
                    candidate_high,
                    rtol=relative_tolerance,
                    atol=absolute_tolerance,
                    equal_nan=False,
                )
            )
        elif not torch.equal(reference_value, candidate_value):
            mathematical = False

    return NumericalEquivalence(
        structural_identity=structural,
        mathematical_equivalence=mathematical and finite,
        bitwise_parameter_identity=bitwise,
        maximum_absolute_parameter_difference=maximum_absolute,
        maximum_relative_parameter_difference=maximum_relative,
        finite_parameters=finite,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


def prediction_identity(reference: list[int] | None, candidate: list[int] | None) -> bool | None:
    if reference is None or candidate is None:
        return None
    return reference == candidate
