"""JSON-serializable median, ridge, and robust linear regression."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fedapfa.utilities.serialization import atomic_write_json, canonical_json


@dataclass(frozen=True)
class CostModel:
    schema_version: int
    target: str
    model_family: str
    feature_order: list[str]
    coefficients: list[float]
    intercept: float
    standardization_means: list[float]
    standardization_scales: list[float]
    target_transformation: str
    fitting_row_hashes: list[str]
    fitting_dataset_identities: list[str]
    fitting_seeds: list[int]
    validation_decision: dict
    software_provenance: dict

    def predict(self, rows: list[dict]) -> np.ndarray:
        if not self.feature_order:
            values = np.repeat(self.intercept, len(rows)).astype(np.float64)
        else:
            matrix = np.asarray(
                [[float(row[name]) for name in self.feature_order] for row in rows], dtype=np.float64
            )
            means = np.asarray(self.standardization_means)
            scales = np.asarray(self.standardization_scales)
            values = self.intercept + ((matrix - means) / scales) @ np.asarray(self.coefficients)
        return np.exp(values) if self.target_transformation == "log" else values

    def record(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        atomic_write_json(path, self.record())

    @classmethod
    def load(cls, path: str | Path) -> CostModel:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _row_hash(row: dict) -> str:
    return hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest()


def _standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales == 0, 1.0, scales)
    return (matrix - means) / scales, means, scales


def fit_regression(
    rows: list[dict],
    target: str,
    features: list[str],
    family: str,
    regularization: float = 0.0,
    target_transformation: str = "identity",
    validation_decision: dict | None = None,
    software_provenance: dict | None = None,
) -> CostModel:
    if not rows:
        raise ValueError("regression fitting rows cannot be empty")
    if "client_id" in features:
        raise ValueError("client ID cannot be a cost-model predictor")
    if target_transformation not in {"identity", "log"}:
        raise ValueError("target transformation must be identity or log")
    response = np.asarray([float(row[target]) for row in rows], dtype=np.float64)
    if not np.isfinite(response).all() or (target_transformation == "log" and np.any(response <= 0)):
        raise ValueError("regression response is incompatible")
    response_fit = np.log(response) if target_transformation == "log" else response
    if family == "median":
        coefficients = means = []
        scales = []
        intercept = float(np.median(response_fit))
    else:
        matrix = np.asarray([[float(row[name]) for name in features] for row in rows], dtype=np.float64)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("regression predictors must be finite")
        standardized, means_array, scales_array = _standardize(matrix)
        design = np.column_stack([np.ones(len(rows)), standardized])
        if family == "ridge":
            penalty = np.eye(design.shape[1]) * float(regularization)
            penalty[0, 0] = 0.0
            solution = np.linalg.pinv(design.T @ design + penalty) @ design.T @ response_fit
        elif family == "robust":
            solution = np.linalg.lstsq(design, response_fit, rcond=None)[0]
            for _ in range(40):
                residual = response_fit - design @ solution
                scale = max(float(np.median(np.abs(residual))) / 0.6744897501960817, 1e-12)
                ratio = np.abs(residual) / (1.345 * scale)
                weights = np.where(ratio <= 1, 1.0, 1.0 / ratio)
                weighted = design * np.sqrt(weights)[:, None]
                response_weighted = response_fit * np.sqrt(weights)
                update = np.linalg.lstsq(weighted, response_weighted, rcond=None)[0]
                if np.max(np.abs(update - solution)) <= 1e-12:
                    solution = update
                    break
                solution = update
        else:
            raise ValueError("regression family must be median, ridge, or robust")
        intercept = float(solution[0])
        coefficients = [float(value) for value in solution[1:]]
        means = [float(value) for value in means_array]
        scales = [float(value) for value in scales_array]
    model = CostModel(
        1,
        target,
        family,
        list(features),
        coefficients,
        intercept,
        means,
        scales,
        target_transformation,
        [_row_hash(row) for row in rows],
        sorted({str(row.get("dataset_identity", row["dataset"])) for row in rows}),
        sorted({int(row["scientific_seed"]) for row in rows}),
        validation_decision or {},
        software_provenance or {},
    )
    predictions = model.predict(rows)
    if not np.isfinite(predictions).all():
        raise RuntimeError("fitted model produced non-finite predictions")
    return model
