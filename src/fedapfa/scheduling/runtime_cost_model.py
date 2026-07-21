"""Strict deployment of the committed event-structure wall-time model."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import EVENT_STRUCTURE_FEATURES


MODEL_CONFIGURATION_KEYS = {
    "name",
    "artifact_path",
    "expected_sha256",
    "schema_version",
    "target",
    "feature_order",
    "fitting_seeds",
    "excluded_evaluation_seed",
    "source_execution_commit",
    "compatible_dataset_models",
    "provenance_artifact_path",
    "expected_provenance_sha256",
    "total_accepted_rows",
    "model_fitting_rows",
    "untouched_evaluation_rows",
}
MODEL_ARTIFACT_KEYS = {
    "schema_version",
    "target",
    "model_family",
    "feature_order",
    "coefficients",
    "intercept",
    "standardization_means",
    "standardization_scales",
    "target_transformation",
    "fitting_row_hashes",
    "fitting_dataset_identities",
    "fitting_seeds",
    "validation_decision",
    "software_provenance",
}
FORBIDDEN_PREDICTOR_FRAGMENTS = (
    "client_id",
    "validation",
    "test",
    "current",
    "duration",
    "cuda",
    "energy",
    "spike_count",
    "spike_rate",
    "label",
    "entropy",
    "history",
    "future",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance_identity(record: dict) -> str:
    selected = {
        "model_name": record["name"],
        "artifact_sha256": record["artifact_sha256"],
        "schema_version": record["schema_version"],
        "target": record["target"],
        "model_family": record["model_family"],
        "feature_order": record["feature_order"],
        "fitting_seeds": record["fitting_seeds"],
        "source_execution_commit": record["source_execution_commit"],
        "provenance_artifact_sha256": record["provenance_artifact_sha256"],
        "total_accepted_rows": record["total_accepted_rows"],
        "model_fitting_rows": record["model_fitting_rows"],
        "untouched_evaluation_rows": record["untouched_evaluation_rows"],
    }
    payload = json.dumps(selected, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_sequence(values, label: str, length: int) -> list[float]:
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"cost model {label} has an incompatible dimension")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"cost model {label} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"cost model {label} must be finite")
        result.append(numeric)
    return result


def _strict_sha256(value, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} SHA-256 is invalid")
    return value


def _row_identity_set(values, label: str, expected_count: int) -> set[str]:
    if not isinstance(values, list) or len(values) != expected_count:
        raise ValueError(f"{label} row count is incompatible")
    identities = set(values)
    if len(identities) != expected_count or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in identities
    ):
        raise ValueError(f"{label} row identities are missing or duplicated")
    return identities


def _validate_row_provenance(
    configuration: dict,
    artifact: dict,
    root: Path,
) -> tuple[Path, str, dict]:
    expected_counts = {
        "total_accepted_rows": 6000,
        "model_fitting_rows": 4000,
        "untouched_evaluation_rows": 2000,
    }
    if any(configuration.get(name) != count for name, count in expected_counts.items()):
        raise ValueError("scheduler model row-count configuration is incompatible")
    fitting_hashes = _row_identity_set(
        artifact.get("fitting_row_hashes"),
        "coefficient-fitting",
        expected_counts["model_fitting_rows"],
    )
    if artifact.get("software_provenance", {}).get("accepted_row_count") != expected_counts["total_accepted_rows"]:
        raise ValueError("scheduler model total accepted-row provenance is incompatible")

    configured_path = Path(configuration["provenance_artifact_path"])
    path = configured_path if configured_path.is_absolute() else root / configured_path
    path = path.resolve()
    if not path.is_file():
        raise ValueError("scheduler model row-provenance artifact is missing")
    expected_hash = _strict_sha256(
        configuration["expected_provenance_sha256"],
        "scheduler model provenance artifact",
    )
    observed_hash = file_sha256(path)
    if observed_hash != expected_hash:
        raise ValueError("scheduler model row-provenance artifact SHA-256 mismatch")
    provenance = json.loads(path.read_text(encoding="utf-8"))
    separation = provenance.get("data_separation", {})
    if (
        provenance.get("schema_version") != 1
        or separation.get("fitting_seeds") != [7, 17]
        or separation.get("evaluation_seed") != 27
        or separation.get("client_grouped_validation") is not True
        or separation.get("prequential_seed_27") is not True
        or separation.get("historical_weight_selection", {}).get("seed_27_used") is not False
    ):
        raise ValueError("scheduler model seed-separation provenance is incompatible")

    joint = provenance.get("settings", {}).get("joint", {})
    fitting = _row_identity_set(joint.get("fitting_row_identities"), "model-selection fitting", 2988)
    selection = _row_identity_set(joint.get("validation_row_identities"), "model-selection validation", 1012)
    evaluation = _row_identity_set(
        joint.get("evaluation_row_identities"),
        "untouched seed-27 evaluation",
        expected_counts["untouched_evaluation_rows"],
    )
    if fitting & selection or fitting & evaluation or selection & evaluation:
        raise ValueError("scheduler model fitting, selection, and seed-27 rows overlap")
    if len(fitting | selection) != expected_counts["model_fitting_rows"]:
        raise ValueError("scheduler model coefficient-fitting collection is incomplete")
    if len(fitting | selection | evaluation) != expected_counts["total_accepted_rows"]:
        raise ValueError("scheduler model accepted-row collection is incomplete")

    runtime_models = joint.get("targets", {}).get("client_wall_time_seconds", {})
    event_selection = runtime_models.get("event_structure", {}).get("selection")
    if (
        not isinstance(event_selection, dict)
        or artifact.get("validation_decision") != event_selection
        or set(event_selection.get("selection_rows", ())) != selection
        or runtime_models.get("event_structure", {}).get("feature_availability") != "before_assignment"
        or provenance.get("scheduler_model") != "event_structure"
    ):
        raise ValueError("scheduler model family, feature, or hyperparameter selection provenance is incompatible")
    non_spike_metrics = {
        name: runtime_models.get(name, {})
        .get("selection", {})
        .get("selected", {})
        .get("validation_metrics", {})
        .get("mean_absolute_error")
        for name in ("size", "event_structure")
    }
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in non_spike_metrics.values()
    ) or non_spike_metrics["event_structure"] > non_spike_metrics["size"]:
        raise ValueError("scheduler feature-set selection was not determined by seed-7/17 validation")
    for name in ("constant", "size", "event_structure", "historical_spike"):
        candidate_selection = runtime_models.get(name, {}).get("selection", {})
        if (
            set(candidate_selection.get("selection_rows", ())) != selection
            or not candidate_selection.get("candidates")
            or candidate_selection.get("selection_metric") != "mean_absolute_error"
        ):
            raise ValueError("scheduler regression-family or hyperparameter selection rows are incompatible")
    historical_candidates = separation["historical_weight_selection"].get("candidates")
    if not isinstance(historical_candidates, list) or not historical_candidates:
        raise ValueError("historical-feature hyperparameter candidates are missing")
    for candidate in historical_candidates:
        candidate_fitting = set(candidate.get("fitting_row_identities", ()))
        candidate_selection = set(candidate.get("validation_row_identities", ()))
        if candidate_fitting != fitting or candidate_selection != selection or (
            candidate_fitting | candidate_selection
        ) & evaluation:
            raise ValueError("seed 27 influenced historical-feature hyperparameter selection")

    record = {
        **expected_counts,
        "coefficient_fitting_row_hashes": len(fitting_hashes),
        "client_grouped_fitting_rows": len(fitting),
        "client_grouped_selection_rows": len(selection),
        "untouched_evaluation_seed": 27,
        "seed_27_overlap_with_fitting_or_selection_rows": 0,
        "seed_27_row_hash_in_coefficient_fitting_rows": False,
        "normalization_uses_seed_27": False,
        "coefficient_fitting_uses_seed_27": False,
        "regression_family_selection_uses_seed_27": False,
        "feature_selection_uses_seed_27": False,
        "hyperparameter_selection_uses_seed_27": False,
        "seed_27_used_for_post_freeze_evaluation": True,
        "provenance_artifact_path": str(path),
        "provenance_artifact_sha256": observed_hash,
    }
    return path, observed_hash, record


@dataclass(frozen=True)
class DeployedLinearCostModel:
    feature_order: list[str]
    coefficients: list[float]
    intercept: float
    standardization_means: list[float]
    standardization_scales: list[float]
    target_transformation: str

    def predict(self, rows: list[dict[str, float]]) -> np.ndarray:
        matrix = np.asarray(
            [[float(row[name]) for name in self.feature_order] for row in rows],
            dtype=np.float64,
        )
        values = self.intercept + (
            (matrix - np.asarray(self.standardization_means)) / np.asarray(self.standardization_scales)
        ) @ np.asarray(self.coefficients)
        return np.exp(values) if self.target_transformation == "log" else values


@dataclass(frozen=True)
class FrozenEventStructureModel:
    """A hash-verified model loaded once and never fitted at runtime."""

    model: DeployedLinearCostModel
    artifact_path: Path
    artifact_sha256: str
    provenance_identity: str
    source_execution_commit: str
    row_provenance: dict

    @classmethod
    def load(
        cls,
        configuration: dict,
        *,
        dataset_name: str,
        model_name: str,
        repository_root: str | Path | None = None,
    ) -> FrozenEventStructureModel:
        if not isinstance(configuration, dict) or set(configuration) != MODEL_CONFIGURATION_KEYS:
            raise ValueError(f"scheduler cost model must contain exactly {sorted(MODEL_CONFIGURATION_KEYS)}")
        if configuration["name"] != "event_structure":
            raise ValueError("scheduler cost model must be event_structure")
        configured_features = configuration["feature_order"]
        if configured_features != list(EVENT_STRUCTURE_FEATURES):
            raise ValueError("scheduler cost-model feature order is incompatible")
        if any(
            fragment in feature.lower() for feature in configured_features for fragment in FORBIDDEN_PREDICTOR_FRAGMENTS
        ):
            raise ValueError("scheduler cost model contains a forbidden predictor")
        compatibility = configuration["compatible_dataset_models"]
        if not isinstance(compatibility, dict) or compatibility.get(dataset_name) != model_name:
            raise ValueError("scheduler cost model is incompatible with the dataset/model identity")

        configured_path = Path(configuration["artifact_path"])
        root = Path.cwd() if repository_root is None else Path(repository_root)
        path = configured_path if configured_path.is_absolute() else root / configured_path
        path = path.resolve()
        if "diagnostic_oracle" in str(path).lower() or not path.is_file():
            raise ValueError("scheduler model artifact is missing or diagnostic-only")
        expected_hash = _strict_sha256(configuration["expected_sha256"], "scheduler model expected")
        observed_hash = file_sha256(path)
        if observed_hash != expected_hash:
            raise ValueError("scheduler model artifact SHA-256 mismatch")

        artifact = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(artifact, dict) or set(artifact) != MODEL_ARTIFACT_KEYS:
            raise ValueError("scheduler model artifact schema fields are incompatible")
        if artifact["schema_version"] != configuration["schema_version"] or artifact["schema_version"] != 1:
            raise ValueError("scheduler model schema version is incompatible")
        if artifact["target"] != configuration["target"] or artifact["target"] != "client_wall_time_seconds":
            raise ValueError("scheduler model target must be client wall time")
        if artifact["model_family"] != "ridge":
            raise ValueError("scheduler model regression family must be ridge")
        if artifact["feature_order"] != configured_features:
            raise ValueError("scheduler model stored feature order is incompatible")
        if artifact["fitting_seeds"] != configuration["fitting_seeds"] or artifact["fitting_seeds"] != [7, 17]:
            raise ValueError("scheduler model fitting seeds must be exactly 7 and 17")
        if configuration["excluded_evaluation_seed"] != 27 or 27 in artifact["fitting_seeds"]:
            raise ValueError("scheduler model seed-27 separation is incompatible")
        if dataset_name not in artifact["fitting_dataset_identities"]:
            raise ValueError("scheduler model was not fitted for the configured dataset identity")

        length = len(EVENT_STRUCTURE_FEATURES)
        coefficients = _finite_sequence(artifact["coefficients"], "coefficients", length)
        means = _finite_sequence(artifact["standardization_means"], "means", length)
        scales = _finite_sequence(artifact["standardization_scales"], "scales", length)
        if any(value <= 0 for value in scales):
            raise ValueError("scheduler model fitting scales must be positive")
        intercept = artifact["intercept"]
        if isinstance(intercept, bool) or not isinstance(intercept, (int, float)) or not math.isfinite(intercept):
            raise ValueError("scheduler model intercept must be finite")
        if artifact["target_transformation"] not in {"identity", "log"}:
            raise ValueError("scheduler model target transformation is incompatible")
        if not isinstance(artifact["fitting_row_hashes"], list) or not artifact["fitting_row_hashes"]:
            raise ValueError("scheduler model fitting-row provenance is missing")
        if any(not isinstance(value, str) or len(value) != 64 for value in artifact["fitting_row_hashes"]):
            raise ValueError("scheduler model fitting-row provenance is incompatible")

        decision = artifact["validation_decision"]
        if (
            not isinstance(decision, dict)
            or decision.get("selection_metric") != "mean_absolute_error"
            or decision.get("selected", {}).get("family") != artifact["model_family"]
            or decision.get("selected", {}).get("target_transformation") != artifact["target_transformation"]
            or not decision.get("selection_rows")
        ):
            raise ValueError("scheduler model selection provenance is incompatible")
        provenance = artifact["software_provenance"]
        source_commit = configuration["source_execution_commit"]
        if (
            not isinstance(provenance, dict)
            or provenance.get("git_commit") != source_commit
            or source_commit != "3ddae173c89125bc69922d80bde5732ed6cd050e"
            or provenance.get("historical_weight_selection", {}).get("seed_27_used") is not False
        ):
            raise ValueError("scheduler model fitting provenance is incompatible")
        if any(
            fragment in feature.lower()
            for feature in artifact["feature_order"]
            for fragment in FORBIDDEN_PREDICTOR_FRAGMENTS
        ):
            raise ValueError("scheduler model artifact contains a forbidden predictor")

        _, provenance_artifact_hash, row_provenance = _validate_row_provenance(
            configuration,
            artifact,
            root,
        )

        model = DeployedLinearCostModel(
            feature_order=list(artifact["feature_order"]),
            coefficients=coefficients,
            intercept=float(intercept),
            standardization_means=means,
            standardization_scales=scales,
            target_transformation=artifact["target_transformation"],
        )
        record = {
            "name": configuration["name"],
            "artifact_sha256": observed_hash,
            "schema_version": artifact["schema_version"],
            "target": artifact["target"],
            "model_family": artifact["model_family"],
            "feature_order": artifact["feature_order"],
            "fitting_seeds": artifact["fitting_seeds"],
            "source_execution_commit": source_commit,
            "provenance_artifact_sha256": provenance_artifact_hash,
            **{
                name: row_provenance[name]
                for name in ("total_accepted_rows", "model_fitting_rows", "untouched_evaluation_rows")
            },
        }
        return cls(model, path, observed_hash, _provenance_identity(record), source_commit, row_provenance)

    def predict(self, features: list[dict[str, float]]) -> list[float]:
        for row in features:
            if set(row) != set(EVENT_STRUCTURE_FEATURES):
                raise ValueError("event-structure predictors must match the strict feature schema")
            for name in EVENT_STRUCTURE_FEATURES:
                value = row[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"event-structure feature {name} must be numeric")
                if not math.isfinite(float(value)):
                    raise ValueError(f"event-structure feature {name} must be finite")
        predictions = self.model.predict(features)
        if not isinstance(predictions, np.ndarray) or predictions.shape != (len(features),):
            raise RuntimeError("scheduler model returned an incompatible prediction shape")
        values = [float(value) for value in predictions]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("scheduler model produced a nonfinite prediction")
        if any(value < 0 for value in values):
            raise ValueError("scheduler model produced a negative duration")
        return values
