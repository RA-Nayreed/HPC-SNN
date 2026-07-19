import numpy as np
import pytest

from fedapfa.cost_estimation.decision import decide_spike_history, ensure_exportable
from fedapfa.cost_estimation.evaluation import regression_metrics
from fedapfa.cost_estimation.regression import CostModel, fit_regression


def _rows(count=30, outlier=False):
    values = []
    for index in range(count):
        response = 3.0 + 2.0 * index
        if outlier and index == count - 1:
            response += 5000
        values.append(
            {
                "dataset": "shd",
                "dataset_identity": "shd-a",
                "scientific_seed": 7,
                "x": float(index),
                "constant": 4.0,
                "target": response,
            }
        )
    return values


def test_ridge_recovers_relation_and_handles_constant_feature():
    rows = _rows()
    model = fit_regression(rows, "target", ["x", "constant"], "ridge", regularization=1e-12)
    assert model.standardization_scales[1] == 1.0
    assert model.predict(rows) == pytest.approx([row["target"] for row in rows], abs=1e-9)
    with pytest.raises(ValueError, match="client ID"):
        fit_regression(rows, "target", ["client_id"], "ridge")


def test_robust_regression_limits_single_extreme_response():
    rows = _rows(outlier=True)
    ridge = fit_regression(rows, "target", ["x"], "ridge")
    robust = fit_regression(rows, "target", ["x"], "robust")
    expected = 3.0 + 2.0 * 10
    assert abs(robust.predict([rows[10]])[0] - expected) < abs(ridge.predict([rows[10]])[0] - expected)


def test_json_round_trip_reproduces_predictions(tmp_path):
    rows = _rows()
    model = fit_regression(rows, "target", ["x"], "ridge", software_provenance={"git": "abc"})
    path = tmp_path / "model.json"
    model.save(path)
    restored = CostModel.load(path)
    assert np.array_equal(model.predict(rows), restored.predict(rows))


def test_exact_metrics_and_positive_percentage_floor():
    metrics = regression_metrics([0.0, 2.0], [1.0, 4.0], percentage_denominator_floor=0.5)
    assert metrics["mean_absolute_error"] == 1.5
    assert metrics["root_mean_squared_error"] == pytest.approx(np.sqrt(2.5))
    assert metrics["median_absolute_percentage_error"] == 1.5
    assert metrics["mean_signed_error"] == 1.5
    assert metrics["sample_count"] == 2
    with pytest.raises(ValueError, match="positive"):
        regression_metrics([1], [1], 0.0)


def _metric(median, p90=0.3, rank=0.8):
    return {
        "median_absolute_error": median,
        "p90_absolute_error": p90,
        "p90_absolute_percentage_error": p90,
        "spearman_rank_correlation": rank,
    }


def test_spike_adoption_and_rejection_decisions():
    accepted = {
        dataset: {
            "historical_spike": _metric(0.9, 0.2, 0.81),
            "strongest_non_spike": _metric(1.0, 0.3, 0.80),
        }
        for dataset in ("shd", "ssc")
    }
    decision = decide_spike_history(accepted, 0.0001, {"historical_spike": 1.0, "strongest_non_spike": 2.0})
    assert decision["decision"] == "spike_history_adopted"
    accepted["ssc"]["historical_spike"] = _metric(0.97, 0.2, 0.81)
    decision = decide_spike_history(accepted, 0.0001, {"historical_spike": 1.0, "strongest_non_spike": 2.0})
    assert decision["decision"] == "spike_history_not_adopted"
    with pytest.raises(ValueError, match="cannot be exported"):
        ensure_exportable("diagnostic_oracle")
