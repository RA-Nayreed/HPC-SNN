"""Resource record schemas, JSON-line loading, and acceptance checks."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from fedapfa.utilities.serialization import atomic_write_json, canonical_json

REQUIRED_RESOURCE_FIELDS = frozenset(
    {
        "dataset",
        "experiment",
        "scientific_seed",
        "communication_round",
        "selected_position",
        "client_id",
        "training_seed",
        "execution_attempt",
        "gpu_uuid",
        "example_count",
        "local_batch_count",
        "total_raw_input_events",
        "mean_sequence_length",
        "median_sequence_length",
        "maximum_sequence_length",
        "total_valid_time_bins",
        "estimated_padded_time_bins",
        "padding_fraction",
        "event_density",
        "represented_class_count",
        "label_entropy",
        "client_wall_time_seconds",
        "data_wait_time_seconds",
        "cuda_event_time_seconds",
        "residual_host_time_seconds",
        "gross_energy_joules",
        "idle_adjusted_energy_joules",
        "energy_sample_count",
        "energy_coverage_seconds",
        "accepted",
    }
)


def read_jsonl(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.is_file():
        return []
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: str | Path, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def resource_row_key(value: dict) -> tuple:
    return (
        value["dataset"],
        int(value["scientific_seed"]),
        int(value["communication_round"]),
        int(value["selected_position"]),
    )


def validate_resource_record(value: dict) -> None:
    missing = REQUIRED_RESOURCE_FIELDS - set(value)
    if missing:
        raise ValueError(f"resource record is missing fields: {sorted(missing)}")
    if not value["accepted"]:
        raise ValueError("client-cost input record is not accepted")
    numeric = [item for item in value.values() if isinstance(item, (int, float))]
    if any(not math.isfinite(float(item)) for item in numeric):
        raise ValueError("resource record contains a non-finite value")
    for name in (
        "client_wall_time_seconds",
        "data_wait_time_seconds",
        "cuda_event_time_seconds",
        "residual_host_time_seconds",
        "gross_energy_joules",
        "idle_adjusted_energy_joules",
    ):
        if float(value[name]) < 0:
            raise ValueError(f"resource record {name} cannot be negative")
    total = sum(
        float(value[name])
        for name in ("data_wait_time_seconds", "cuda_event_time_seconds", "residual_host_time_seconds")
    )
    if not math.isclose(total, float(value["client_wall_time_seconds"]), rel_tol=0.0, abs_tol=2e-6):
        raise ValueError("resource record timing components do not reconcile")


def record_hash(value: dict) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_measurement_acceptance(
    path: str | Path,
    *,
    execution_completion: bool,
    measurement_completeness: bool,
    energy_completeness: bool,
    scientific_hypothesis_outcome: str,
    findings: list[str],
) -> dict:
    value = {
        "schema_version": 1,
        "execution_completion": execution_completion,
        "measurement_completeness": measurement_completeness,
        "energy_completeness": energy_completeness,
        "scientific_hypothesis_outcome": scientific_hypothesis_outcome,
        "accepted": execution_completion and measurement_completeness and energy_completeness,
        "validation_findings": list(findings),
    }
    atomic_write_json(path, value)
    return value
