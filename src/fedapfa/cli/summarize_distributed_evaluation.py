"""Summarize workload-paired single-node distributed FedAvg executions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from fedapfa.configuration import (
    FEDERATED_SEEDS,
    load_device_capacity_manifest,
    load_distributed_evaluation_manifest,
)
from fedapfa.metrics.parallel_efficiency import execution_speedup, parallel_efficiency


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _jsonl(path: Path) -> list[dict]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(value, dict) for value in values):
        raise ValueError(f"{path} contains a non-object record")
    return values


def _number(value, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _stats(values: list[float]) -> dict[str, float]:
    if len(values) != 3:
        raise ValueError("distributed evaluation statistics require seeds 7, 17, and 27")
    return {
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _discover(root: Path, names: set[str]) -> dict[tuple[str, int], list[Path]]:
    grouped: dict[tuple[str, int], list[Path]] = defaultdict(list)
    if not root.is_dir():
        return grouped
    for path in sorted(root.iterdir()):
        config_path = path / "resolved_config.yaml"
        if not path.is_dir() or not config_path.is_file():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if config.get("name") in names and config.get("seed") in FEDERATED_SEEDS:
            grouped[(config["name"], int(config["seed"]))].append(path)
    return grouped


def _allocated_gpus(value: str) -> int:
    candidates = []
    for item in value.split(","):
        key, separator, count = item.partition("=")
        if separator and (key == "gres/gpu" or key.startswith("gres/gpu:")):
            candidates.append(int(count))
    if len(candidates) != 1 or candidates[0] <= 0:
        raise ValueError("Slurm AllocTRES must contain one positive GPU allocation")
    return candidates[0]


def _read_slurm_accounting(path: str | Path) -> dict[str, dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="|"))
    required = {"JobIDRaw", "State", "ExitCode", "ElapsedRaw", "AllocTRES"}
    if not rows or any(not required.issubset(row) for row in rows):
        raise ValueError("Slurm accounting fields are incomplete")
    records = {}
    for row in rows:
        job_id = row["JobIDRaw"]
        if not job_id or "." in job_id or "_" not in job_id:
            continue
        if job_id in records:
            raise ValueError(f"Slurm accounting contains duplicate job {job_id}")
        elapsed = int(row["ElapsedRaw"])
        devices = _allocated_gpus(row["AllocTRES"])
        records[job_id] = {
            "job_id": job_id,
            "state": row["State"],
            "exit_code": row["ExitCode"],
            "elapsed_seconds": elapsed,
            "allocated_gpu_count": devices,
            "allocated_gpu_hours": elapsed * devices / 3600,
            "allocation": row["AllocTRES"],
        }
    return records


def _selected_state(path: Path, final: dict) -> dict[str, torch.Tensor]:
    artifact = final.get("selected_checkpoint_artifact")
    if not isinstance(artifact, str) or not artifact:
        raise ValueError("selected checkpoint artifact is missing")
    checkpoint = torch.load(path / artifact, map_location="cpu", weights_only=False)
    state = checkpoint.get("global_model_state")
    if not isinstance(state, dict) or not state:
        raise ValueError("selected checkpoint model state is missing")
    return {name: value.detach().cpu() for name, value in state.items()}


def _validated_utilization(task, measurements: dict, resume_count: int) -> dict:
    utilization = measurements.get("gpu_utilization")
    if not isinstance(utilization, dict):
        raise ValueError("physical-device utilization record is missing")
    if utilization.get("sampling_interval_seconds") != 2:
        raise ValueError("physical-device utilization interval is incompatible")
    if utilization.get("scope") != "latest_execution_attempt":
        raise ValueError("physical-device utilization scope is incompatible")
    if utilization.get("execution_attempt") != resume_count + 1:
        raise ValueError("physical-device utilization attempt identity is incompatible")
    expected_devices = {
        str(value) for value in range(task.config["parallel_execution"]["device_count"])
    }
    by_device = utilization.get("by_device_index")
    if not isinstance(by_device, dict) or set(by_device) != expected_devices:
        raise ValueError("physical-device utilization coverage is incompatible")
    total_samples = utilization.get("sample_count")
    if not isinstance(total_samples, int) or isinstance(total_samples, bool) or total_samples <= 0:
        raise ValueError("physical-device utilization sample count is incompatible")
    device_samples = 0
    for device_index, device_record in by_device.items():
        if not isinstance(device_record, dict):
            raise ValueError(f"physical-device utilization record {device_index} is incompatible")
        sample_count = device_record.get("sample_count")
        if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count <= 0:
            raise ValueError(f"physical-device utilization sample count {device_index} is incompatible")
        device_samples += sample_count
        values = [
            _number(device_record.get(key), f"physical-device utilization {device_index} {key}")
            for key in ("minimum_percent", "mean_percent", "maximum_percent")
        ]
        if not 0.0 <= values[0] <= values[1] <= values[2] <= 100.0:
            raise ValueError(f"physical-device utilization values {device_index} are incompatible")
    if device_samples != total_samples:
        raise ValueError("physical-device utilization sample counts do not sum exactly")
    values = [
        _number(utilization.get(key), f"physical-device utilization {key}")
        for key in ("minimum_percent", "mean_percent", "maximum_percent")
    ]
    if not 0.0 <= values[0] <= values[1] <= values[2] <= 100.0:
        raise ValueError("physical-device utilization aggregate values are incompatible")
    return utilization


def _load_run(task, path: Path) -> dict:
    acceptance = _json(path / "acceptance.json")
    final = _json(path / "final_metrics.json")
    official = _json(path / "official_test_metrics.json")
    measurements = _json(path / "execution_measurements.json")
    rounds = _jsonl(path / "round_metrics.jsonl")
    clients = _jsonl(path / "client_metrics.jsonl")
    parallel = task.config["parallel_execution"]
    expected_clients = task.config["federated"]["rounds"] * task.config["federated"]["clients_per_round"]
    if acceptance.get("completed") is not True or final.get("completed") is not True:
        raise ValueError("execution did not pass completion checks")
    expected_status = "equivalence_not_established" if task.dataset == "cifar10" else "not_claimed"
    if acceptance.get("scientific_status") != expected_status:
        raise ValueError("scientific status is incompatible with the workload protocol")
    if acceptance.get("parallel_execution") != parallel:
        raise ValueError("parallel execution record is incompatible with the manifest")
    if len(rounds) != 100 or [value.get("round_number") for value in rounds] != list(range(1, 101)):
        raise ValueError("communication-round records are incomplete")
    if len(clients) != expected_clients:
        raise ValueError("client update record count is incomplete")
    if measurements.get("completed") is not True or len(measurements.get("rounds", [])) != 100:
        raise ValueError("execution measurements are incomplete")
    if final.get("data_protocol", {}).get("official_test_access_count") != 1:
        raise ValueError("official test access count must be one")
    if (
        official.get("access_count") != 1
        or official.get("evaluation_completed") is not True
        or official.get("evaluated_after_model_selection") is not True
        or official.get("monitored_during_rounds") is not False
        or official.get("complete_split") is not True
    ):
        raise ValueError("official test record is incompatible")
    official_metrics = official.get("metrics")
    if not isinstance(official_metrics, dict) or official_metrics != final.get("test"):
        raise ValueError("official test metrics differ from the final record")
    official_examples = official_metrics.get("examples")
    if not isinstance(official_examples, int) or isinstance(official_examples, bool) or official_examples <= 0:
        raise ValueError("official test example count must be positive")
    predictions = official_metrics.get("predictions")
    if predictions is not None and (
        not isinstance(predictions, list)
        or not predictions
        or any(not isinstance(value, int) or isinstance(value, bool) for value in predictions)
    ):
        raise ValueError("official test prediction record is incompatible")
    timing_names = (
        "model_distribution_time_seconds",
        "result_collection_time_seconds",
        "aggregation_time_seconds",
        "validation_time_seconds",
        "checkpoint_time_seconds",
    )
    timing_totals = {
        name: sum(_number(value[name], name) for value in measurements["rounds"]) for name in timing_names
    }
    busy_by_rank = {
        str(rank): sum(
            _number(value["process_busy_time_seconds"][str(rank)], "process busy time")
            for value in measurements["rounds"]
        )
        for rank in range(parallel["process_count"])
    }
    peak_allocated = {
        str(rank): max(
            int(value["peak_cuda_memory_bytes_by_process_rank"][str(rank)])
            for value in measurements["rounds"]
        )
        for rank in range(parallel["process_count"])
    }
    peak_reserved = {
        str(rank): max(
            int(value["peak_cuda_reserved_bytes_by_process_rank"][str(rank)])
            for value in measurements["rounds"]
        )
        for rank in range(parallel["process_count"])
    }
    total_round_time = sum(_number(value["total_round_time_seconds"], "round time") for value in rounds)
    parallel_client_time = sum(
        _number(value["parallel_client_training_wall_time_seconds"], "parallel client time") for value in rounds
    )
    resume_count = int(measurements.get("resume_count", 0))
    if resume_count < 0:
        raise ValueError("distributed resumption count is incompatible")
    utilization = _validated_utilization(task, measurements, resume_count)
    utilization_mean = (
        None
        if resume_count != 0
        else _number(utilization["mean_percent"], "GPU utilization")
    )
    resource_allocations = measurements.get("resource_allocations")
    if not isinstance(resource_allocations, list) or not resource_allocations:
        resource_allocations = [measurements.get("resource_allocation", {})]
    return {
        "workload": task.dataset,
        "experiment": task.experiment,
        "seed": task.seed,
        "device_count": parallel["device_count"],
        "client_processes_per_device": parallel["client_processes_per_device"],
        "process_count": parallel["process_count"],
        "control_backend": parallel["control_backend"],
        "cuda_process_service": parallel["cuda_process_service"],
        "run_directory": str(path),
        "official_test_accuracy": _number(final["test"]["accuracy"], "official-test accuracy"),
        "official_test_macro_f1": _number(final["test"]["macro_f1"], "official-test macro-F1"),
        "best_validation_accuracy": (
            None
            if final.get("best_validation_accuracy") is None
            else _number(final["best_validation_accuracy"], "best validation accuracy")
        ),
        "selected_round": int(final["selected_round"]),
        "checkpoint_selection": final["checkpoint_selection"],
        "total_runtime_seconds": total_round_time,
        "total_round_time_seconds": total_round_time,
        "mean_round_time_seconds": total_round_time / 100,
        "parallel_client_training_wall_time_seconds": parallel_client_time,
        **timing_totals,
        "process_busy_time_seconds_by_rank": busy_by_rank,
        "process_load_imbalance": max(
            _number(value["process_load_imbalance"], "load imbalance") for value in measurements["rounds"]
        ),
        "maximum_process_peak_cuda_memory_bytes": max(peak_allocated.values()),
        "maximum_process_peak_cuda_reserved_bytes": max(peak_reserved.values()),
        "peak_cuda_memory_bytes_by_process_rank": peak_allocated,
        "peak_cuda_reserved_bytes_by_process_rank": peak_reserved,
        "gpu_utilization_percent": utilization_mean,
        "gpu_utilization_record": utilization,
        "gpu_utilization_scope": (
            "complete_uninterrupted_execution" if utilization_mean is not None else "unavailable_for_comparison"
        ),
        "resume_count": resume_count,
        "logical_communication_bytes": int(final["logical_communication"]["cumulative_total_bytes"]),
        "execution_data_movement_bytes": int(final["execution_data_movement"]["total_bytes"]),
        "split_id": final["split_id"],
        "partition_id": final["partition_id"],
        "model_initialization_id": final["model_initialization_id"],
        "scientific_identity": final["scientific_identity"],
        "execution_identity": final["execution_identity"],
        "selected_client_order": [value["selected_client_ids"] for value in rounds],
        "client_training_seeds": [value["resolved_training_seed"] for value in clients],
        "client_example_counts": [value["example_count"] for value in clients],
        "aggregation_weighting": final["aggregation_weighting"],
        "official_test_access_count": final["data_protocol"]["official_test_access_count"],
        "official_test_examples": official_examples,
        "official_test_dataset_identity": official.get("dataset_identity"),
        "git_commit": acceptance["git_commit"],
        "scientific_status": acceptance["scientific_status"],
        "resource_allocation": measurements.get("resource_allocation", {}),
        "resource_allocations": resource_allocations,
        "_predictions": predictions,
        "_selected_state": _selected_state(path, final),
    }


def _parameter_difference(
    reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]
) -> tuple[float, float]:
    if set(reference) != set(candidate):
        raise ValueError("selected checkpoint parameter keys differ")
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for name in sorted(reference):
        left = reference[name]
        right = candidate[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"selected checkpoint tensor {name} differs in shape or dtype")
        if left.is_floating_point() or left.is_complex():
            absolute = (right - left).abs()
            maximum_absolute = max(maximum_absolute, float(absolute.max()) if absolute.numel() else 0.0)
            denominator = left.abs().clamp_min(torch.finfo(left.dtype).eps)
            relative = absolute / denominator
            maximum_relative = max(maximum_relative, float(relative.max()) if relative.numel() else 0.0)
        elif not torch.equal(left, right):
            left_numeric = left.to(torch.float64)
            right_numeric = right.to(torch.float64)
            absolute = (right_numeric - left_numeric).abs()
            maximum_absolute = max(maximum_absolute, float(absolute.max()))
            denominator = left_numeric.abs().clamp_min(torch.finfo(torch.float64).eps)
            maximum_relative = max(maximum_relative, float((absolute / denominator).max()))
    return maximum_absolute, maximum_relative


def _comparison(reference: dict, run: dict) -> dict:
    structural_fields = (
        "scientific_identity",
        "split_id",
        "partition_id",
        "model_initialization_id",
        "selected_client_order",
        "client_training_seeds",
        "client_example_counts",
        "aggregation_weighting",
        "logical_communication_bytes",
        "checkpoint_selection",
        "official_test_access_count",
        "official_test_examples",
        "official_test_dataset_identity",
        "git_commit",
    )
    differences = [field for field in structural_fields if run[field] != reference[field]]
    maximum_absolute, maximum_relative = _parameter_difference(
        reference["_selected_state"], run["_selected_state"]
    )
    reference_predictions = reference["_predictions"]
    run_predictions = run["_predictions"]
    prediction_records_compatible = (reference_predictions is None) == (run_predictions is None)
    prediction_agreement = None
    if reference_predictions is not None and run_predictions is not None:
        prediction_records_compatible = len(reference_predictions) == len(run_predictions)
        if prediction_records_compatible:
            prediction_agreement = sum(
                left == right for left, right in zip(reference_predictions, run_predictions, strict=True)
            ) / len(reference_predictions)
    exact_parameters = maximum_absolute == 0.0 and maximum_relative == 0.0
    selected_round_equal = run["selected_round"] == reference["selected_round"]
    accuracy_difference = run["official_test_accuracy"] - reference["official_test_accuracy"]
    macro_f1_difference = run["official_test_macro_f1"] - reference["official_test_macro_f1"]
    exact_evaluation = (
        accuracy_difference == 0.0
        and macro_f1_difference == 0.0
        and prediction_records_compatible
        and (prediction_agreement is None or prediction_agreement == 1.0)
    )
    return {
        "structural_identity_status": "exact" if not differences else "different",
        "structural_differences": differences,
        "numerical_equivalence_status": (
            "exact"
            if not differences and exact_parameters and selected_round_equal and exact_evaluation
            else "difference_observed"
        ),
        "maximum_absolute_parameter_difference": maximum_absolute,
        "maximum_relative_parameter_difference": maximum_relative,
        "selected_round_equal": selected_round_equal,
        "official_test_accuracy_difference": accuracy_difference,
        "official_test_macro_f1_difference": macro_f1_difference,
        "prediction_records_compatible": prediction_records_compatible,
        "prediction_agreement": prediction_agreement,
    }


def _manifest_tasks(path: str | Path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    collection = value.get("collection")
    if collection == "distributed_evaluation":
        return load_distributed_evaluation_manifest(path), collection
    if collection == "device_capacity_evaluation":
        return load_device_capacity_manifest(path), collection
    raise ValueError("summary manifest collection is incompatible")


def summarize_distributed_evaluation(
    manifest: str | Path,
    runs_root: str | Path,
    output_dir: str | Path,
    slurm_accounting: str | Path | None = None,
) -> dict:
    tasks, collection = _manifest_tasks(manifest)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    discovered = _discover(Path(runs_root), {task.experiment for task in tasks})
    findings: list[str] = []
    runs: list[dict] = []
    for task in tasks:
        candidates = discovered.get((task.experiment, task.seed), [])
        if len(candidates) != 1:
            findings.append(f"{task.experiment} seed {task.seed}: expected one execution, found {len(candidates)}")
            continue
        try:
            runs.append(_load_run(task, candidates[0]))
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            findings.append(f"{task.experiment} seed {task.seed}: {error}")

    accounting = _read_slurm_accounting(slurm_accounting) if slurm_accounting is not None else {}
    if slurm_accounting is not None and not accounting:
        findings.append("Slurm accounting contains no array-task records")
    if accounting:
        used_accounting: set[str] = set()
        for run in runs:
            allocation_records = []
            for allocation in run["resource_allocations"]:
                array_job = allocation.get("array_job_id")
                array_task = allocation.get("array_task_id")
                job_id = f"{array_job}_{array_task}" if array_job and array_task is not None else None
                record = accounting.get(job_id)
                if record is None:
                    findings.append(
                        f"{run['experiment']} seed {run['seed']}: Slurm accounting record is missing"
                    )
                    continue
                used_accounting.add(job_id)
                allocation_records.append(record)
                if record["allocated_gpu_count"] != run["device_count"]:
                    findings.append(f"{run['experiment']} seed {run['seed']}: allocated GPU count differs")
            if allocation_records:
                completed_allocation = allocation_records[-1]
                if (
                    completed_allocation["state"] != "COMPLETED"
                    or completed_allocation["exit_code"] != "0:0"
                ):
                    findings.append(
                        f"{run['experiment']} seed {run['seed']}: "
                        "completing Slurm allocation was not successful"
                    )
                run["slurm_accounting"] = allocation_records
                run["allocation_states"] = [value["state"] for value in allocation_records]
                run["noncompleted_allocation_count"] = sum(
                    value["state"] != "COMPLETED" or value["exit_code"] != "0:0"
                    for value in allocation_records[:-1]
                )
                run["allocated_gpu_hours"] = sum(
                    value["allocated_gpu_hours"] for value in allocation_records
                )
                run["slurm_elapsed_seconds"] = sum(
                    value["elapsed_seconds"] for value in allocation_records
                )
        for job_id in sorted(set(accounting).difference(used_accounting)):
            findings.append(f"Slurm accounting contains unexpected array task {job_id}")

    workload_summaries = []
    comparisons = []
    for workload in dict.fromkeys(task.dataset for task in tasks):
        workload_runs = [value for value in runs if value["workload"] == workload]
        by_key = {
            (value["device_count"], value["client_processes_per_device"], value["seed"]): value
            for value in workload_runs
        }
        treatments = []
        workload_topologies = {
            (
                task.config["parallel_execution"]["device_count"],
                task.config["parallel_execution"]["client_processes_per_device"],
            )
            for task in tasks
            if task.dataset == workload
        }
        for device_count, processes_per_device in sorted(workload_topologies):
            treatment_runs = [
                by_key[(device_count, processes_per_device, seed)]
                for seed in FEDERATED_SEEDS
                if (device_count, processes_per_device, seed) in by_key
            ]
            paired = []
            for run in treatment_runs:
                reference = by_key.get((1, 1, run["seed"]))
                if reference is None:
                    findings.append(f"{workload} seed {run['seed']}: one-GPU execution reference is unavailable")
                    continue
                comparison = _comparison(reference, run)
                speedup = execution_speedup(reference["total_round_time_seconds"], run["total_round_time_seconds"])
                comparison.update(
                    {
                        "workload": workload,
                        "seed": run["seed"],
                        "device_count": device_count,
                        "client_processes_per_device": run["client_processes_per_device"],
                        "speedup": speedup,
                        "parallel_efficiency": parallel_efficiency(speedup, run["process_count"]),
                    }
                )
                comparisons.append(comparison)
                paired.append(comparison)
            metrics = {}
            if len(treatment_runs) == 3:
                for key in (
                    "official_test_accuracy",
                    "official_test_macro_f1",
                    "total_runtime_seconds",
                    "total_round_time_seconds",
                    "mean_round_time_seconds",
                    "parallel_client_training_wall_time_seconds",
                    "aggregation_time_seconds",
                    "validation_time_seconds",
                    "maximum_process_peak_cuda_memory_bytes",
                    "maximum_process_peak_cuda_reserved_bytes",
                    "process_load_imbalance",
                ):
                    metrics[key] = _stats([float(value[key]) for value in treatment_runs])
                validation_values = [value["best_validation_accuracy"] for value in treatment_runs]
                if all(value is not None for value in validation_values):
                    metrics["best_validation_accuracy"] = _stats([float(value) for value in validation_values])
                if all(value.get("allocated_gpu_hours") is not None for value in treatment_runs):
                    metrics["allocated_gpu_hours"] = _stats(
                        [float(value["allocated_gpu_hours"]) for value in treatment_runs]
                    )
                    metrics["slurm_elapsed_seconds"] = _stats(
                        [float(value["slurm_elapsed_seconds"]) for value in treatment_runs]
                    )
                utilization_values = [value["gpu_utilization_percent"] for value in treatment_runs]
                if all(isinstance(value, (int, float)) for value in utilization_values):
                    metrics["gpu_utilization_percent"] = _stats(
                        [float(value) for value in utilization_values]
                    )
            paired_metrics = {}
            if len(paired) == 3:
                for key in (
                    "speedup",
                    "parallel_efficiency",
                    "maximum_absolute_parameter_difference",
                    "maximum_relative_parameter_difference",
                    "official_test_accuracy_difference",
                    "official_test_macro_f1_difference",
                ):
                    paired_metrics[key] = _stats([float(value[key]) for value in paired])
                prediction_values = [value["prediction_agreement"] for value in paired]
                if all(isinstance(value, (int, float)) for value in prediction_values):
                    paired_metrics["prediction_agreement"] = _stats(
                        [float(value) for value in prediction_values]
                    )
            public_runs = [
                {key: value for key, value in run.items() if not key.startswith("_")}
                for run in treatment_runs
            ]
            treatments.append(
                {
                    "device_count": device_count,
                    "client_processes_per_device": processes_per_device,
                    "process_count": device_count * processes_per_device,
                    "completed_seed_count": len(treatment_runs),
                    "completed": len(treatment_runs) == 3,
                    "metrics": metrics,
                    "paired_to_one_gpu": paired_metrics,
                    "structural_identity_status": (
                        "exact"
                        if paired and all(value["structural_identity_status"] == "exact" for value in paired)
                        else "unavailable_or_different"
                    ),
                    "numerical_equivalence_status": (
                        "exact"
                        if paired and all(value["numerical_equivalence_status"] == "exact" for value in paired)
                        else "unavailable_or_difference_observed"
                    ),
                    "selected_round_status": (
                        "exact"
                        if paired and all(value["selected_round_equal"] for value in paired)
                        else "unavailable_or_different"
                    ),
                    "runs": public_runs,
                }
            )
        workload_summaries.append({"workload": workload, "treatments": treatments})

    summary = {
        "schema_version": 2,
        "valid": len(runs) == len(tasks) and not findings,
        "validation_findings": findings,
        "collection": collection,
        "expected_task_count": len(tasks),
        "completed_task_count": len(runs),
        "required_seeds": list(FEDERATED_SEEDS),
        "reference_execution": "one_gpu_distributed_path_within_each_workload",
        "cross_workload_pooling": False,
        "logical_communication_definition": "federated model downloads and uploads only",
        "execution_data_movement_reported_separately": True,
        "runtime_definition": "sum_of_communication_round_total_durations",
        "slurm_accounting_provided": slurm_accounting is not None,
        "workloads": workload_summaries,
        "paired_records": comparisons,
    }
    (output / "distributed_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_csv(output / "distributed_evaluation_summary.csv", workload_summaries)
    _write_markdown(output / "distributed_evaluation_summary.md", summary)
    return summary


def _write_csv(path: Path, workloads: list[dict]) -> None:
    rows = []
    for workload in workloads:
        for treatment in workload["treatments"]:
            row = {
                "workload": workload["workload"],
                "device_count": treatment["device_count"],
                "client_processes_per_device": treatment["client_processes_per_device"],
                "process_count": treatment["process_count"],
                "completed_seed_count": treatment["completed_seed_count"],
                "structural_identity_status": treatment["structural_identity_status"],
                "numerical_equivalence_status": treatment["numerical_equivalence_status"],
            }
            for section in ("metrics", "paired_to_one_gpu"):
                for metric, values in treatment[section].items():
                    for statistic, value in values.items():
                        row[f"{section}_{metric}_{statistic}"] = value
            rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _formatted(values: dict) -> str:
    return "n/a" if not values else f"{values['mean']:.6g} ± {values['sample_standard_deviation']:.6g}"


def _write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Single-node distributed FedAvg evaluation",
        "",
        f"Status: **{'valid' if summary['valid'] else 'invalid'}**",
        "",
        "Each workload uses its own one-GPU distributed execution as the timing and numerical reference.",
    ]
    for workload in summary["workloads"]:
        lines.extend(
            [
                "",
                f"## {workload['workload'].upper()}",
                "",
                "| GPUs | Client processes/GPU | Seeds | Test accuracy | Macro-F1 | Peak allocation (bytes) | "
                "GPU utilization (%) | Load imbalance | Structural | Numerical |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for treatment in workload["treatments"]:
            metrics = treatment["metrics"]
            lines.append(
                f"| {treatment['device_count']} | {treatment['client_processes_per_device']} | "
                f"{treatment['completed_seed_count']}/3 | "
                f"{_formatted(metrics.get('official_test_accuracy', {}))} | "
                f"{_formatted(metrics.get('official_test_macro_f1', {}))} | "
                f"{_formatted(metrics.get('maximum_process_peak_cuda_memory_bytes', {}))} | "
                f"{_formatted(metrics.get('gpu_utilization_percent', {}))} | "
                f"{_formatted(metrics.get('process_load_imbalance', {}))} | "
                f"{treatment['structural_identity_status']} | {treatment['numerical_equivalence_status']} |"
            )
        lines.extend(
            [
                "",
                "| GPUs | Client processes/GPU | Total runtime (s) | Round time (s) | Client wall time (s) | "
                "Aggregation (s) | Validation (s) | Speedup | Efficiency |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for treatment in workload["treatments"]:
            metrics = treatment["metrics"]
            paired = treatment["paired_to_one_gpu"]
            lines.append(
                f"| {treatment['device_count']} | {treatment['client_processes_per_device']} | "
                f"{_formatted(metrics.get('total_runtime_seconds', {}))} | "
                f"{_formatted(metrics.get('mean_round_time_seconds', {}))} | "
                f"{_formatted(metrics.get('parallel_client_training_wall_time_seconds', {}))} | "
                f"{_formatted(metrics.get('aggregation_time_seconds', {}))} | "
                f"{_formatted(metrics.get('validation_time_seconds', {}))} | "
                f"{_formatted(paired.get('speedup', {}))} | "
                f"{_formatted(paired.get('parallel_efficiency', {}))} |"
            )
    lines.extend(
        [
            "",
            "Total runtime is the sum of recorded communication-round total durations. Speedup and parallel "
            "efficiency are paired within each workload. Profiling traces, when enabled, add overhead and are "
            "not ordinary runtime evidence. Internal execution movement is separate from logical federated "
            "communication.",
        ]
    )
    if summary["validation_findings"]:
        lines.extend(["", "## Validation findings", ""])
        lines.extend(f"- {value}" for value in summary["validation_findings"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize single-node distributed FedAvg evaluations.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slurm-accounting")
    args = parser.parse_args()
    summary = summarize_distributed_evaluation(
        args.manifest, args.runs_root, args.output_dir, args.slurm_accounting
    )
    if not summary["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
