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

    def run_once(measured):
        random.random()
        np.random.random()
        torch.rand(1)
        with torch.no_grad():
            model.weight.add_(1)
        return 1.01 if measured else 1.0, {"weight": model.weight.detach().clone()}, 10, ["GPU-a"]

    result = calibrate_measurement(run_once, model)
    assert result["passed"]
    assert result["median_relative_overhead"] == pytest.approx(0.01)
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
