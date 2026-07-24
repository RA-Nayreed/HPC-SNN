"""Strict construction of the accepted client-cost table."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import yaml

from fedapfa.configuration.resource_measurement import validate_resource_measurement_config
from fedapfa.measurement.features import FEATURE_AVAILABILITY
from fedapfa.measurement.records import read_jsonl, resource_row_key, validate_resource_record
from fedapfa.utilities.serialization import atomic_write_json, atomic_write_text


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
    lines = []
    if fields:
        from io import StringIO

        stream = StringIO()
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
        lines.append(stream.getvalue())
    atomic_write_text(path, "".join(lines))


def validate_accepted_run(run_dir: Path, strict_collection: bool = True) -> tuple[list[dict], dict]:
    required = {
        "measurement_acceptance.json",
        "measurement_config.json",
        "calibration_reference.json",
        "idle_power.json",
        "device_samples.jsonl",
        "execution_intervals.jsonl",
        "client_resource_records.jsonl",
        "excluded_intervals.jsonl",
        "final_metrics.json",
        "git.json",
        "resolved_config.yaml",
        "split.json",
        "partition.json",
        "model_initialization.json",
        "execution_measurements.json",
        "execution_provenance.json",
    }
    if strict_collection:
        required.add("client_metrics.jsonl")
    missing = sorted(name for name in required if not (run_dir / name).is_file())
    if missing:
        raise ValueError(f"run {run_dir} is missing artifacts: {missing}")
    acceptance = json.loads((run_dir / "measurement_acceptance.json").read_text(encoding="utf-8"))
    if not acceptance.get("accepted") or any(
        acceptance.get(name) is not True
        for name in ("execution_completion", "measurement_completeness", "energy_completeness")
    ):
        raise ValueError(f"run {run_dir} did not pass measurement acceptance")
    if acceptance.get("validation_findings") or acceptance.get("sampling_error_count") != 0:
        raise ValueError(f"run {run_dir} contains measurement findings")
    calibration = json.loads((run_dir / "calibration_reference.json").read_text(encoding="utf-8"))
    calibration_references = calibration.get("attempts")
    if not isinstance(calibration_references, list):
        calibration_references = [calibration]
    calibration_attempts = set()
    if not calibration_references:
        raise ValueError(f"run {run_dir} lacks passing calibration")
    for reference in calibration_references:
        calibration_artifact = reference.get("artifact", {})
        if (
            not calibration_artifact.get("passed")
            or calibration_artifact.get("official_test_access_count") != 0
            or calibration_artifact.get("sampling_errors")
        ):
            raise ValueError(f"run {run_dir} lacks passing calibration")
        if reference.get("execution_attempt") is not None:
            calibration_attempts.add(int(reference["execution_attempt"]))
    final = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    if not final.get("completed") or not final.get("accepted"):
        raise ValueError(f"run {run_dir} did not complete the established training acceptance")
    protocol = final.get("data_protocol", {})
    if protocol.get("official_test_access_count") != 1 or protocol.get("official_test_monitored_during_training"):
        raise ValueError(f"run {run_dir} violates official-test isolation")
    execution_identity = final.get("execution_identity", {})
    stored_execution_identity = json.loads(
        (run_dir / "execution_provenance.json").read_text(encoding="utf-8")
    )
    if stored_execution_identity != execution_identity:
        raise ValueError(f"run {run_dir} has inconsistent execution provenance")
    parallel = final.get("parallel_execution", {})
    if (
        parallel.get("node_count") != 1
        or parallel.get("device_count") != 1
        or parallel.get("process_count") != 1
        or parallel.get("client_processes_per_device") != 1
        or parallel.get("cuda_process_service") != "none"
        or parallel.get("control_backend") != "nccl"
    ):
        raise ValueError(f"run {run_dir} has an incompatible process topology")
    git_record = json.loads((run_dir / "git.json").read_text(encoding="utf-8"))
    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    partition = json.loads((run_dir / "partition.json").read_text(encoding="utf-8"))
    initialization = json.loads((run_dir / "model_initialization.json").read_text(encoding="utf-8"))
    resolved_config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    if strict_collection:
        validate_resource_measurement_config(resolved_config)
    model_configuration_identity = hashlib.sha256(
        json.dumps(resolved_config["model"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    commit = git_record.get("commit")
    if not commit or execution_identity.get("git_commit") != commit:
        raise ValueError(f"run {run_dir} has incompatible Git provenance")
    if execution_identity.get("configuration_id") != final.get("configuration_id"):
        raise ValueError(f"run {run_dir} has incompatible configuration provenance")
    hardware = execution_identity.get("hardware_allocation", {})
    device_names = hardware.get("device_names", [])
    if strict_collection and (
        hardware.get("visible_device_count") != 1
        or len(device_names) != 1
        or "GH200" not in str(device_names[0])
    ):
        raise ValueError(f"run {run_dir} lacks the required GH200 hardware identity")
    execution_measurements = json.loads(
        (run_dir / "execution_measurements.json").read_text(encoding="utf-8")
    )
    allocation = execution_measurements.get("resource_allocation", {})
    if not allocation.get("job_id"):
        raise ValueError(f"run {run_dir} lacks a Slurm allocation identity")
    resource_allocations = execution_measurements.get("resource_allocations", [allocation])
    allocation_ids = sorted(
        {
            str(value["job_id"])
            for value in resource_allocations
            if value.get("job_id")
        }
    )
    if not allocation_ids:
        raise ValueError(f"run {run_dir} lacks Slurm allocation provenance")
    records = read_jsonl(run_dir / "client_resource_records.jsonl")
    if strict_collection and len(records) != 1000:
        raise ValueError(f"run {run_dir} must contain exactly 1,000 client-resource records")
    client_sizes = {
        str(value["client_id"]): int(value["size"]) for value in partition.get("clients", [])
    }
    if strict_collection:
        scientific_records = read_jsonl(run_dir / "client_metrics.jsonl")
        if len(scientific_records) != 1000:
            raise ValueError(f"run {run_dir} must contain exactly 1,000 established client records")
        scientific_by_key = {}
        for value in scientific_records:
            key = (int(value["round_number"]), int(value["selected_position"]))
            if key in scientific_by_key:
                raise ValueError(f"run {run_dir} duplicates an established client identity")
            scientific_by_key[key] = value
        resource_by_key = {}
        for value in records:
            key = (int(value["communication_round"]), int(value["selected_position"]))
            if key in resource_by_key:
                raise ValueError(f"run {run_dir} duplicates a client-resource identity")
            resource_by_key[key] = value
        if set(scientific_by_key) != set(resource_by_key):
            raise ValueError(f"run {run_dir} is missing a selected client-resource mapping")
        for key, scientific in scientific_by_key.items():
            resource = resource_by_key[key]
            if (
                str(resource["client_id"]) != str(scientific["client_id"])
                or int(resource["training_seed"]) != int(scientific["resolved_training_seed"])
            ):
                raise ValueError(f"run {run_dir} has a client-resource training identity mismatch")
        interval_records = read_jsonl(run_dir / "execution_intervals.jsonl")
        interval_counts = Counter(value["interval_id"] for value in interval_records)
        duplicated_intervals = [interval_id for interval_id, count in interval_counts.items() if count != 1]
        if duplicated_intervals:
            raise ValueError(f"run {run_dir} duplicates an interval identity")
        intervals_by_id = {}
        for value in interval_records:
            intervals_by_id[value["interval_id"]] = value
        row_interval_ids = {str(value["interval_id"]) for value in records}
        if len(row_interval_ids) != len(records):
            raise ValueError(f"run {run_dir} maps more than one row to a client interval")
        for interval_id in row_interval_ids:
            interval = intervals_by_id.get(interval_id)
            if (
                interval_counts[interval_id] != 1
                or interval is None
                or interval.get("category") != "client_training"
                or interval.get("accepted") is not True
            ):
                raise ValueError(f"run {run_dir} has an incompatible client interval mapping")
        sample_uuids_by_attempt = {}
        sampling_intervals = set()
        sample_attempts = set()
        with (run_dir / "device_samples.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                if sample.get("sampling_error_status") is not None:
                    raise ValueError(f"run {run_dir} contains a device sampling error")
                attempt = int(sample.get("execution_attempt", -1))
                sample_uuids_by_attempt.setdefault(attempt, set()).add(
                    str(sample.get("gpu_uuid"))
                )
                sampling_intervals.add(int(sample.get("configured_interval_ms", -1)))
                sample_attempts.add(attempt)
        if (
            any(len(values) != 1 for values in sample_uuids_by_attempt.values())
            or sampling_intervals != {100}
            or not sample_attempts
        ):
            raise ValueError(f"run {run_dir} has an incompatible device sample trace")
        row_attempts = {int(value["execution_attempt"]) for value in records}
        if not row_attempts.issubset(sample_attempts):
            raise ValueError(f"run {run_dir} mixes client and power trace identities")
        for value in records:
            attempt = int(value["execution_attempt"])
            if str(value["gpu_uuid"]) not in sample_uuids_by_attempt[attempt]:
                raise ValueError(f"run {run_dir} mixes client and power trace UUIDs")
        if not row_attempts.issubset(calibration_attempts):
            raise ValueError(f"run {run_dir} lacks attempt-specific calibration provenance")
        idle = json.loads((run_dir / "idle_power.json").read_text(encoding="utf-8"))
        idle_records = idle.get("attempts", [])
        if not any("idle_after" in value for value in idle_records):
            raise ValueError(f"run {run_dir} lacks a post-evaluation idle interval")
        for value in idle_records:
            for category in ("idle_before", "idle_after"):
                if category in value:
                    duration = (
                        int(value[category]["end_ns"]) - int(value[category]["start_ns"])
                    ) / 1_000_000_000
                    if duration < 29.9:
                        raise ValueError(f"run {run_dir} has a shortened idle interval")
        idle_attempts = {
            int(value["execution_attempt"])
            for value in idle_records
            if "combined_median_power_watts" in value
        }
        if not row_attempts.issubset(idle_attempts):
            raise ValueError(f"run {run_dir} lacks an attempt-specific idle baseline")
        for value in idle_records:
            attempt = int(value["execution_attempt"])
            if (
                attempt in row_attempts
                and str(value.get("gpu_uuid")) not in sample_uuids_by_attempt[attempt]
            ):
                raise ValueError(f"run {run_dir} has an idle UUID mismatch")
    for row in records:
        validate_resource_record(row)
        if row.get("feature_source_scope") != "client_training_indices":
            raise ValueError(f"run {run_dir} has an incompatible feature source")
        if row.get("validation_indices_in_features") or row.get("official_test_indices_in_features"):
            raise ValueError(f"run {run_dir} contains validation or test feature leakage")
        if row.get("client_id") not in client_sizes or int(row["example_count"]) != client_sizes[row["client_id"]]:
            raise ValueError(f"run {run_dir} has a client identity or size mismatch")
        if int(row["energy_sample_count"]) < 2 or not math.isclose(
            float(row["energy_coverage_seconds"]),
            float(row["client_wall_time_seconds"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"run {run_dir} has incomplete power coverage")
        if float(row["idle_adjusted_energy_joules"]) > float(row["gross_energy_joules"]):
            raise ValueError(f"run {run_dir} has incompatible energy targets")
        identity_expectations = {
            "split_id": split.get("split_id"),
            "partition_id": partition.get("partition_id"),
            "model_initialization_id": initialization.get("model_initialization_id"),
            "model_configuration_identity": model_configuration_identity,
            "git_commit": commit,
        }
        if any(row.get(name) != expected for name, expected in identity_expectations.items()):
            raise ValueError(f"run {run_dir} has incompatible row provenance")
        if row.get("model_identity") != final.get("model_class") or int(row.get("parameter_count", -1)) != int(
            final.get("parameter_count", -2)
        ):
            raise ValueError(f"run {run_dir} has an incompatible model identity")
        if strict_collection:
            interval = intervals_by_id[row["interval_id"]]
            identity = interval.get("identity", {})
            if (
                int(interval["execution_attempt"]) != int(row["execution_attempt"])
                or interval["gpu_uuid"] != row["gpu_uuid"]
                or str(identity.get("dataset")) != str(row["dataset"])
                or str(identity.get("experiment")) != str(row["experiment"])
                or int(identity.get("scientific_seed", -1)) != int(row["scientific_seed"])
                or int(identity.get("communication_round", -1)) != int(row["communication_round"])
                or int(identity.get("selected_position", -1)) != int(row["selected_position"])
                or str(identity.get("client_id")) != str(row["client_id"])
                or int(identity.get("training_seed", -1)) != int(row["training_seed"])
                or int(identity.get("execution_attempt", -1)) != int(row["execution_attempt"])
                or str(identity.get("gpu_uuid")) != str(row["gpu_uuid"])
            ):
                raise ValueError(f"run {run_dir} has an incompatible interval identity")
    input_hashes = {name: _file_hash(run_dir / name) for name in sorted(required)}
    provenance = {
        "run_directory": str(run_dir.resolve()),
        "input_sha256": input_hashes,
        "git_commit": commit,
        "configuration_id": final.get("configuration_id"),
        "dataset": records[0]["dataset"] if records else None,
        "scientific_seed": records[0]["scientific_seed"] if records else None,
        "sampling_interval_ms": records[0].get("sampling_interval_ms") if records else None,
        "slurm_allocation_id": allocation.get("job_id"),
        "slurm_allocation_ids": allocation_ids,
    }
    return records, provenance


def build_client_cost_dataset(
    run_directories: list[str | Path],
    result_root: str | Path,
    expected_rows: int = 6000,
) -> dict:
    """Read accepted records only and write a deterministically ordered analysis table."""

    root = Path(result_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    provenance = []
    exclusions = []
    for value in sorted({Path(item).resolve() for item in run_directories}):
        try:
            run_rows, run_provenance = validate_accepted_run(value, expected_rows == 6000)
            rows.extend(run_rows)
            provenance.append(run_provenance)
        except (ValueError, OSError, json.JSONDecodeError) as error:
            exclusions.append({"run_directory": str(value), "reason": str(error)})
    keys = [resource_row_key(value) for value in rows]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        exclusions.extend({"row_key": list(key), "reason": "duplicate_row_identity"} for key in duplicates)
    rows.sort(key=resource_row_key)
    expected_keys = {
        (dataset, seed, round_number, position)
        for dataset in ("shd", "ssc")
        for seed in (7, 17, 27)
        for round_number in range(1, 101)
        for position in range(10)
    }
    missing = sorted(expected_keys - set(keys)) if expected_rows == 6000 else []
    if missing:
        exclusions.extend({"row_key": list(key), "reason": "missing_row_identity"} for key in missing)
    commits = {value["git_commit"] for value in provenance}
    intervals = {value["sampling_interval_ms"] for value in provenance}
    if len(commits) != 1 or None in commits:
        exclusions.append({"reason": "mismatched_git_provenance"})
    if intervals != {100}:
        exclusions.append({"reason": "mixed_sampling_intervals", "values": sorted(intervals, key=str)})
    if expected_rows == 6000:
        run_identities = {
            (value["dataset"], int(value["scientific_seed"])) for value in provenance
        }
        expected_run_identities = {
            (dataset, seed) for dataset in ("shd", "ssc") for seed in (7, 17, 27)
        }
        if run_identities != expected_run_identities or len(provenance) != 6:
            exclusions.append({"reason": "scientific_run_matrix_mismatch"})
        per_dataset_models = {
            dataset: {
                (
                    row.get("model_identity"),
                    row.get("model_configuration_identity"),
                    row.get("parameter_count"),
                )
                for row in rows
                if row["dataset"] == dataset
            }
            for dataset in ("shd", "ssc")
        }
        if any(len(values) != 1 for values in per_dataset_models.values()):
            exclusions.append({"reason": "incompatible_model_identity"})
    for row in rows:
        numeric = [value for value in row.values() if isinstance(value, (int, float))]
        if any(not math.isfinite(float(value)) for value in numeric):
            exclusions.append({"row_key": list(resource_row_key(row)), "reason": "non_finite_value"})
    atomic_write_json(root / "excluded_rows.json", {"schema_version": 1, "records": exclusions})
    if len(rows) != expected_rows:
        exclusions.append({"reason": "accepted_row_count", "expected": expected_rows, "observed": len(rows)})
    if exclusions:
        atomic_write_json(root / "excluded_rows.json", {"schema_version": 1, "records": exclusions})
        raise ValueError(f"client-cost dataset validation failed with {len(exclusions)} findings")
    _write_csv(root / "client_cost_data.csv", rows)
    scalar_fields = sorted(
        {key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))}
    )
    properties = {}
    for field in scalar_fields:
        types = set()
        for row in rows:
            value = row.get(field)
            if value is None:
                types.add("null")
            elif isinstance(value, bool):
                types.add("boolean")
            elif isinstance(value, int):
                types.add("integer")
            elif isinstance(value, float):
                types.add("number")
            else:
                types.add("string")
        ordered_types = sorted(types)
        properties[field] = {"type": ordered_types[0] if len(ordered_types) == 1 else ordered_types}
    availability_by_field = {
        field: category
        for category, fields in FEATURE_AVAILABILITY.items()
        for field in fields
    }
    feature_dictionary = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_version": 1,
        "title": "Accepted federated SNN client cost row",
        "type": "object",
        "required": scalar_fields,
        "properties": properties,
        "additionalProperties": False,
        "feature_availability": FEATURE_AVAILABILITY,
        "feature_dictionary": {
            field: {
                "availability": availability_by_field.get(field, "identity_or_provenance"),
                "scheduler_input": availability_by_field.get(field)
                in {"before_any_client_execution", "after_previous_observations"}
                and field != "client_id",
            }
            for field in scalar_fields
        },
        "targets": {
            "client_wall_time_seconds": "primary scheduling target",
            "cuda_event_time_seconds": "summed CUDA-event duration",
            "gross_energy_joules": "primary device-energy target",
            "idle_adjusted_energy_joules": "separate dynamic-energy target",
        },
        "scheduler_exclusions": ["client_id", "current_execution_spike_counts", "current_execution_spike_rates"],
    }
    atomic_write_json(root / "client_cost_schema.json", feature_dictionary)
    provenance_record = {
        "schema_version": 1,
        "accepted_row_count": len(rows),
        "deterministic_order": ["dataset", "scientific_seed", "communication_round", "selected_position"],
        "input_runs": provenance,
        "git_commit": next(iter(commits)),
        "sampling_interval_ms": 100,
    }
    atomic_write_json(root / "client_cost_provenance.json", provenance_record)
    return {"rows": rows, "schema": feature_dictionary, "provenance": provenance_record}
