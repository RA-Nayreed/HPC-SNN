import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from fedapfa.measurement.calibration import calibrate_measurement


def test_calibration_alternates_order_restores_state_and_passes_declared_gates():
    torch.manual_seed(13)
    model = nn.Linear(2, 1)
    initial = copy.deepcopy(model.state_dict())
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    calls = []

    def run_once(measured):
        calls.append(measured)
        random.random()
        np.random.random()
        torch.rand(1)
        with torch.no_grad():
            model.weight.add_(1)
        return 1.01 if measured else 1.0, {"weight": model.weight.detach().clone()}, 10, ["GPU-a"]

    result = calibrate_measurement(run_once, model)
    assert result["passed"]
    assert result["median_relative_overhead"] == pytest.approx(0.01)
    assert calls[:2] == [True, False]
    assert len(calls) == 22
    assert len(result["observations"]) == 10
    assert result["warm_up_policy"] == {
        "execution_order": ["measured", "unmeasured"],
        "measured_executions": 1,
        "unmeasured_executions": 1,
        "included_in_paired_observations": False,
        "included_in_overhead_statistic": False,
        "state_restored_before_each_execution": True,
    }
    assert [value["first_condition"] for value in result["observations"]] == [
        "measured" if index % 2 == 0 else "unmeasured" for index in range(10)
    ]
    assert all(torch.equal(model.state_dict()[name], value) for name, value in initial.items())
    assert random.getstate() == python_state
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_state[0]
    assert np.array_equal(restored_numpy[1], numpy_state[1])
    assert restored_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_calibration_rejects_update_difference_and_insufficient_samples():
    model = nn.Linear(1, 1)

    def run_once(measured):
        return 1.0, {"weight": torch.tensor([float(measured)])}, 2, ["GPU-a"]

    result = calibrate_measurement(run_once, model)
    assert not result["passed"]
    assert "sample_count_coverage_failed" in result["validation_findings"]
    assert "measured_update_identity_failed" in result["validation_findings"]


def test_calibration_excludes_warm_up_and_preserves_two_percent_gate():
    model = nn.Linear(1, 1)
    calls = 0

    def run_once(measured):
        nonlocal calls
        calls += 1
        if calls <= 2:
            duration = 100.0 if measured else 0.01
        else:
            duration = 1.021 if measured else 1.0
        return duration, {"weight": torch.tensor([1.0])}, 10, ["GPU-a"]

    result = calibrate_measurement(run_once, model)
    assert len(result["observations"]) == 10
    assert result["median_relative_overhead"] == pytest.approx(0.021)
    assert not result["passed"]
    assert "median_runtime_overhead_exceeded" in result["validation_findings"]
