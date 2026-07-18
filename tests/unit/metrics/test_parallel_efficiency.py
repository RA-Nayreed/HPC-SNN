import pytest

from fedapfa.metrics.parallel_efficiency import (
    example_throughput,
    execution_speedup,
    parallel_efficiency,
)


def test_execution_scaling_metrics_use_paired_duration_boundaries():
    speedup = execution_speedup(120.0, 50.0)
    assert speedup == 2.4
    assert parallel_efficiency(speedup, 4) == 0.6
    assert example_throughput(10000, 50.0) == 200.0


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: execution_speedup(0, 1), "durations"),
        (lambda: execution_speedup(1, float("nan")), "durations"),
        (lambda: parallel_efficiency(1, 0), "parallel width"),
        (lambda: example_throughput(0, 1), "examples"),
    ],
)
def test_execution_scaling_metrics_reject_invalid_inputs(call, message):
    with pytest.raises(ValueError, match=message):
        call()
