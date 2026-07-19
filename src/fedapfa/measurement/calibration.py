"""Paired instrumentation calibration with exact state restoration."""

from __future__ import annotations

import copy
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch

from fedapfa.federated.randomness import global_rng_state, restore_global_rng_state


@dataclass(frozen=True)
class CalibrationObservation:
    repetition: int
    first_condition: str
    measured_seconds: float
    unmeasured_seconds: float
    relative_overhead: float
    measured_sample_count: int
    updates_identical: bool


def calibrate_measurement(
    run_once: Callable[[bool], tuple[float, dict[str, torch.Tensor], int, list[str]]],
    model: torch.nn.Module,
    repetitions: int = 10,
    maximum_median_overhead: float = 0.02,
    minimum_samples: int = 10,
    minimum_sample_fraction: float = 0.9,
    device: torch.device | None = None,
) -> dict:
    """Warm both paths, then alternate recorded pairs with exact state restoration."""

    if repetitions < 10:
        raise ValueError("calibration requires at least ten paired repetitions")
    initial_state = copy.deepcopy(model.state_dict())
    initial_rng = global_rng_state(device if device is not None and device.type == "cuda" else None)
    observations: list[CalibrationObservation] = []
    errors: list[str] = []
    uuids: set[str] = set()

    def restore_initial_state() -> None:
        model.load_state_dict(initial_state, strict=True)
        restore_global_rng_state(
            initial_rng,
            device if device is not None and device.type == "cuda" else None,
        )

    try:
        for measured in (True, False):
            restore_initial_state()
            run_once(measured)
        for repetition in range(repetitions):
            values = {}
            order = [True, False] if repetition % 2 == 0 else [False, True]
            for measured in order:
                restore_initial_state()
                duration, update, sample_count, observed_uuids = run_once(measured)
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError("calibration duration must be finite and positive")
                values[measured] = (duration, update, sample_count)
                if measured:
                    uuids.update(observed_uuids)
            measured_value = values[True]
            unmeasured_value = values[False]
            identical = set(measured_value[1]) == set(unmeasured_value[1]) and all(
                torch.equal(measured_value[1][name], unmeasured_value[1][name])
                for name in measured_value[1]
            )
            observations.append(
                CalibrationObservation(
                    repetition,
                    "measured" if order[0] else "unmeasured",
                    measured_value[0],
                    unmeasured_value[0],
                    (measured_value[0] - unmeasured_value[0]) / unmeasured_value[0],
                    int(measured_value[2]),
                    identical,
                )
            )
    finally:
        restore_initial_state()
    overheads = [value.relative_overhead for value in observations]
    sample_fraction = sum(value.measured_sample_count >= minimum_samples for value in observations) / repetitions
    median_overhead = statistics.median(overheads)
    if median_overhead > maximum_median_overhead:
        errors.append("median_runtime_overhead_exceeded")
    if sample_fraction < minimum_sample_fraction:
        errors.append("sample_count_coverage_failed")
    if len(uuids) != 1:
        errors.append("gpu_uuid_count_failed")
    if not all(value.updates_identical for value in observations):
        errors.append("measured_update_identity_failed")
    return {
        "schema_version": 1,
        "warm_up_policy": {
            "execution_order": ["measured", "unmeasured"],
            "measured_executions": 1,
            "unmeasured_executions": 1,
            "included_in_paired_observations": False,
            "included_in_overhead_statistic": False,
            "state_restored_before_each_execution": True,
        },
        "paired_repetitions": repetitions,
        "observations": [value.__dict__ for value in observations],
        "median_relative_overhead": median_overhead,
        "minimum_samples_per_client": minimum_samples,
        "sample_coverage_fraction": sample_fraction,
        "gpu_uuids": sorted(uuids),
        "sampling_errors": [],
        "updates_numerically_identical": all(value.updates_identical for value in observations),
        "official_test_access_count": 0,
        "passed": not errors,
        "validation_findings": errors,
    }
