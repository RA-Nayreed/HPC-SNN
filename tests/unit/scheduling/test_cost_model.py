import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fedapfa.configuration import load_evaluation_config
from fedapfa.scheduling.base import EVENT_STRUCTURE_FEATURES
from fedapfa.scheduling.runtime import SchedulerRuntime
from fedapfa.scheduling.runtime_cost_model import (
    DeployedLinearCostModel,
    FrozenEventStructureModel,
)


CONFIG = "experiments/scheduling_evaluation/shd/lif_fedavg_round_robin.yaml"


def _configuration():
    return copy.deepcopy(load_evaluation_config(CONFIG)["scheduler"]["cost_model"])


def _rewritten_artifact(tmp_path, change):
    configuration = _configuration()
    artifact = json.loads(Path(configuration["artifact_path"]).read_text())
    change(artifact)
    path = tmp_path / "cost_model.json"
    payload = json.dumps(artifact, sort_keys=True).encode()
    path.write_bytes(payload)
    configuration["artifact_path"] = str(path)
    configuration["expected_sha256"] = hashlib.sha256(payload).hexdigest()
    return configuration


def test_frozen_model_matches_stored_linear_transformation_exactly():
    frozen = FrozenEventStructureModel.load(_configuration(), dataset_name="shd", model_name="lif_2layer")
    row = dict(zip(EVENT_STRUCTURE_FEATURES, frozen.model.standardization_means, strict=True))
    expected = frozen.model.intercept + sum(
        ((row[name] - mean) / scale) * coefficient
        for name, mean, scale, coefficient in zip(
            frozen.model.feature_order,
            frozen.model.standardization_means,
            frozen.model.standardization_scales,
            frozen.model.coefficients,
            strict=True,
        )
    )
    if frozen.model.target_transformation == "log":
        expected = float(np.exp(expected))
    assert frozen.predict([row]) == [expected]


def test_model_rejects_missing_nonfinite_and_negative_predictions():
    frozen = FrozenEventStructureModel.load(_configuration(), dataset_name="shd", model_name="lif_2layer")
    row = {name: 1.0 for name in EVENT_STRUCTURE_FEATURES}
    missing = dict(row)
    missing.pop(EVENT_STRUCTURE_FEATURES[0])
    with pytest.raises(ValueError, match="strict feature schema"):
        frozen.predict([missing])
    row[EVENT_STRUCTURE_FEATURES[0]] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        frozen.predict([row])
    negative = FrozenEventStructureModel(
        DeployedLinearCostModel(list(EVENT_STRUCTURE_FEATURES), [0.0] * 10, -1.0, [0.0] * 10, [1.0] * 10, "identity"),
        Path("model.json"),
        "0" * 64,
        "identity",
        "commit",
        {},
    )
    with pytest.raises(ValueError, match="negative"):
        negative.predict([{name: 0.0 for name in EVENT_STRUCTURE_FEATURES}])


def test_model_rejects_hash_target_oracle_and_forbidden_predictors(tmp_path):
    configuration = _configuration()
    configuration["expected_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        FrozenEventStructureModel.load(configuration, dataset_name="shd", model_name="lif_2layer")

    wrong_target = _rewritten_artifact(tmp_path, lambda value: value.update(target="energy"))
    with pytest.raises(ValueError, match="target"):
        FrozenEventStructureModel.load(wrong_target, dataset_name="shd", model_name="lif_2layer")

    diagnostic = _configuration()
    diagnostic["artifact_path"] = str(tmp_path / "diagnostic_oracle_model.json")
    with pytest.raises(ValueError, match="diagnostic"):
        FrozenEventStructureModel.load(diagnostic, dataset_name="shd", model_name="lif_2layer")

    for forbidden in ("client_id", "current_round_duration"):
        changed = _configuration()
        changed["feature_order"][0] = forbidden
        with pytest.raises(ValueError, match="feature order|forbidden"):
            FrozenEventStructureModel.load(changed, dataset_name="shd", model_name="lif_2layer")


def test_scheduler_runtime_loads_model_once_and_records_schema():
    config = load_evaluation_config(CONFIG)
    runtime = SchedulerRuntime(config, object())
    first = runtime.schedule(["a", "b"], 1, 4)
    second = runtime.schedule(["b", "a"], 2, 4)
    assert runtime.model_load_count == 1
    assert first.strategy == second.strategy == "round_robin"
    assert first.feature_availability == {name: False for name in EVENT_STRUCTURE_FEATURES}
    assert first.metadata_serialized_bytes == 0


def test_frozen_model_proves_fitting_and_untouched_evaluation_row_separation():
    frozen = FrozenEventStructureModel.load(_configuration(), dataset_name="shd", model_name="lif_2layer")
    provenance = frozen.row_provenance
    assert provenance["total_accepted_rows"] == 6000
    assert provenance["model_fitting_rows"] == 4000
    assert provenance["coefficient_fitting_row_hashes"] == 4000
    assert provenance["client_grouped_fitting_rows"] == 2988
    assert provenance["client_grouped_selection_rows"] == 1012
    assert provenance["untouched_evaluation_rows"] == 2000
    assert provenance["untouched_evaluation_seed"] == 27
    assert provenance["seed_27_overlap_with_fitting_or_selection_rows"] == 0
    assert provenance["seed_27_row_hash_in_coefficient_fitting_rows"] is False
    for operation in (
        "normalization",
        "coefficient_fitting",
        "regression_family_selection",
        "feature_selection",
        "hyperparameter_selection",
    ):
        assert provenance[f"{operation}_uses_seed_27"] is False
    assert provenance["seed_27_used_for_post_freeze_evaluation"] is True


def test_frozen_model_rejects_row_count_and_provenance_hash_drift():
    changed = _configuration()
    changed["model_fitting_rows"] = 6000
    with pytest.raises(ValueError, match="row-count"):
        FrozenEventStructureModel.load(changed, dataset_name="shd", model_name="lif_2layer")
    changed = _configuration()
    changed["expected_provenance_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance artifact SHA-256 mismatch"):
        FrozenEventStructureModel.load(changed, dataset_name="shd", model_name="lif_2layer")
