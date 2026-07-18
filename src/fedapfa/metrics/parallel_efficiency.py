"""Execution-time scaling metrics for paired parallel-execution treatments."""

from __future__ import annotations

import math


def execution_speedup(reference_seconds: float, treatment_seconds: float) -> float:
    reference = float(reference_seconds)
    treatment = float(treatment_seconds)
    if not math.isfinite(reference) or not math.isfinite(treatment) or reference <= 0 or treatment <= 0:
        raise ValueError("execution durations must be finite and positive")
    return reference / treatment


def parallel_efficiency(speedup: float, parallel_width: int) -> float:
    value = float(speedup)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("speedup must be finite and positive")
    if not isinstance(parallel_width, int) or isinstance(parallel_width, bool) or parallel_width <= 0:
        raise ValueError("parallel width must be a positive integer")
    return value / parallel_width


def example_throughput(examples: int, seconds: float) -> float:
    if not isinstance(examples, int) or isinstance(examples, bool) or examples <= 0:
        raise ValueError("examples must be a positive integer")
    duration = float(seconds)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    return examples / duration
