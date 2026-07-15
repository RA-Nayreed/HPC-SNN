import copy
import json

import pytest

from fedapfa.configuration import load_config
from fedapfa.training.acceptance import evaluate_acceptance


def _case(tmp_path, config_path="experiments/centralized/shd/lif_independent_evaluation.yaml"):
    config = load_config(config_path)
    config["training"]["epochs"] = 1
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "best_validation.pt").write_bytes(b"best")
    (run / "checkpoints" / "last.pt").write_bytes(b"last")
    (run / "training.log").write_text("trained\n", encoding="utf-8")
    record = {
        "epoch": 0,
        "train": {"loss": 0.5, "accuracy": 0.75},
        "validation": {"loss": 0.4, "accuracy": 0.8},
    }
    (run / "metrics.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (run / "git.json").write_text(json.dumps({"commit": "abc"}), encoding="utf-8")
    final = {
        "test": {"loss": 0.3, "accuracy": 0.79, "spike_rates": {"layer1": 0.1, "layer2": 0.2}},
        "model_class": config["acceptance"]["expected_model_class"],
    }
    metadata = {
        "complete_dataset_used": True,
        "official_test_accessed": True,
        "official_test_monitored_during_training": config["protocol"] == "published_protocol",
        "official_test_evaluated_after_model_selection": True,
        "official_test_role": "final_test_only",
        "official_test_examples": 100,
    }
    termination = {
        "reason": "epochs_completed",
        "last_epoch": 0,
        "configured_epochs": 1,
        "epochs_without_improvement": 0,
        "early_stop_documented": False,
    }
    return config, run, final, metadata, termination


def test_scientific_completion_is_not_unconditional_and_reference_null_is_not_claimed(tmp_path):
    config, run, final, metadata, termination = _case(tmp_path)
    result = evaluate_acceptance(config, run, final, metadata, termination)
    assert result["completed"]
    assert result["scientific_status"] == "not_claimed"
    (run / "checkpoints" / "last.pt").unlink()
    result = evaluate_acceptance(config, run, final, metadata, termination)
    assert not result["completed"]
    assert any("last.pt" in failure for failure in result["completion_failures"])


def test_nan_and_inf_metrics_fail_completion(tmp_path):
    config, run, final, metadata, termination = _case(tmp_path)
    final["test"]["loss"] = float("nan")
    result = evaluate_acceptance(config, run, final, metadata, termination)
    assert not result["completed"]
    assert any("NaN or Inf" in failure for failure in result["completion_failures"])
    final["test"]["loss"] = 0.3
    (run / "metrics.jsonl").write_text(
        json.dumps({"epoch": 0, "train": {"loss": float("inf"), "accuracy": 0.5}}) + "\n",
        encoding="utf-8",
    )
    result = evaluate_acceptance(config, run, final, metadata, termination)
    assert not result["completed"]


def test_missing_official_test_metrics_fail_completion(tmp_path):
    config, run, final, metadata, termination = _case(tmp_path)
    final.pop("test")
    metadata["official_test_accessed"] = False
    result = evaluate_acceptance(config, run, final, metadata, termination)
    assert not result["completed"]
    assert result["achieved_test_accuracy"] is None
    assert any("official test" in failure for failure in result["completion_failures"])


def test_configured_reference_and_tolerance_pass_or_fail_correctly(tmp_path):
    config, run, final, metadata, termination = _case(
        tmp_path, "experiments/centralized/shd/pfa_public_published_protocol.yaml"
    )
    config = copy.deepcopy(config)
    config["acceptance"]["reference_test_accuracy"] = 0.80
    config["acceptance"]["absolute_tolerance"] = 0.02
    final["test"]["accuracy"] = 0.79
    passed = evaluate_acceptance(config, run, final, metadata, termination)
    assert passed["scientific_status"] == "passed"
    assert passed["absolute_accuracy_difference"] == pytest.approx(0.01)
    final["test"]["accuracy"] = 0.70
    failed = evaluate_acceptance(config, run, final, metadata, termination)
    assert failed["scientific_status"] == "failed"
    assert failed["absolute_accuracy_difference"] == pytest.approx(0.1)
