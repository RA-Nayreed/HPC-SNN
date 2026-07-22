"""Calibrate node-local multi-GPU measurement on the resolved execution topology."""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from fedapfa.configuration import load_comparative_evaluation_config
from fedapfa.distributed.process_context import (
    close_process_context,
    initialize_process_context,
    node_process_group,
)
from fedapfa.federated.client import train_client
from fedapfa.federated.communication_accounting import model_payload_bytes
from fedapfa.federated.randomness import derive_seed
from fedapfa.federated.workload import prepare_federated_execution_workload
from fedapfa.measurement.calibration import calibrate_measurement
from fedapfa.measurement.multi_gpu_energy import NodeNvmlProcessSampler, merge_node_telemetry
from fedapfa.utilities.git_metadata import git_metadata
from fedapfa.utilities.serialization import atomic_write_json

from .calibrate_resource_measurement import _CalibrationHook


def _aggregate(records: list[dict], config: dict, telemetry_files: list[str]) -> dict:
    requirements = config["calibration_requirements"]
    observations = [
        {"process_rank": rank, **observation}
        for rank, record in enumerate(records)
        for observation in record["observations"]
    ]
    overheads = [float(value["relative_overhead"]) for value in observations]
    findings = sorted({finding for record in records for finding in record.get("validation_findings", [])})
    median_overhead = statistics.median(overheads)
    sample_coverage = min(float(value["sample_coverage_fraction"]) for value in records)
    updates_identical = all(value["updates_numerically_identical"] for value in records)
    gpu_uuids = sorted({uuid for value in records for uuid in value["gpu_uuids"]})
    if median_overhead > float(requirements["maximum_median_runtime_overhead_fraction"]):
        findings.append("median_runtime_overhead_exceeded")
    if sample_coverage < float(requirements["minimum_interval_coverage_fraction"]):
        findings.append("sample_count_coverage_failed")
    if not updates_identical:
        findings.append("measured_update_identity_failed")
    if len(gpu_uuids) != int(requirements["device_count"]):
        findings.append("gpu_uuid_count_failed")
    findings = sorted(set(findings))
    return {
        "schema_version": 1,
        "warm_up_policy": records[0]["warm_up_policy"],
        "paired_repetitions": int(requirements["paired_repetitions"]),
        "alternating_order": True,
        "observations": observations,
        "per_process_records": records,
        "median_relative_overhead": median_overhead,
        "sample_coverage_fraction": sample_coverage,
        "gpu_uuids": gpu_uuids,
        "node_count": int(requirements["node_count"]),
        "device_count": int(requirements["device_count"]),
        "process_count": int(requirements["process_count"]),
        "sampler_topology": requirements["sampler_topology"],
        "sampling_interval_ms": int(requirements["sampling_interval_ms"]),
        "execution_commit": git_metadata()["commit"],
        "sampling_errors": [],
        "updates_numerically_identical": updates_identical,
        "official_test_access_count": 0,
        "training_data_only": True,
        "telemetry_files": telemetry_files,
        "validation_findings": findings,
        "passed": not findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate comparative node-local NVML measurement.")
    parser.add_argument("config")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    config = load_comparative_evaluation_config(args.config)
    config["dataset"]["root"] = str(Path(args.data_root).resolve())
    config["seed"] = args.seed
    requirements = config["calibration_requirements"]
    context = initialize_process_context(config["parallel_execution"])
    root = Path(args.result_root).resolve()
    try:
        status = [None]
        if context.is_coordinator:
            if root.exists():
                status[0] = "calibration result root already exists"
            else:
                root.mkdir(parents=True)
        dist.broadcast_object_list(status, src=0, device=context.control_device)
        if status[0] is not None:
            raise FileExistsError(status[0])
        dist.barrier()
        workload = prepare_federated_execution_workload(config, coordinator=context.is_coordinator)
        model = workload.model_factory(config).to(context.device)
        bundle = workload.data
        client_id = sorted(bundle.client_ids)[context.rank]
        dataset = bundle.client_dataset(client_id)
        training_seed = derive_seed(config["seed"], config["seed_streams"]["client_training"], 1, client_id)
        payload = model_payload_bytes(model.state_dict())
        mappings = [None for _ in range(context.world_size)]
        dist.all_gather_object(
            mappings,
            {
                "rank": context.rank,
                "node_rank": context.node_rank,
                "host": context.host,
                "gpu_uuid": context.gpu_uuid,
                "gpu_uuid_raw": context.gpu_uuid_raw,
            },
        )
        mappings = [value for value in mappings if value is not None]
        node_mapping = {
            next(value["host"] for value in mappings if value["node_rank"] == node_rank): [
                value["gpu_uuid"] for value in mappings if value["node_rank"] == node_rank
            ]
            for node_rank in range(context.node_count)
        }
        measured_index = 0
        telemetry_files: list[str] = []
        allocation_identity = os.environ.get("SLURM_JOB_ID", "calibration")

        def run_once(measured: bool):
            nonlocal measured_index
            sampler = None
            current_index = measured_index
            if measured:
                measured_index += 1
                if context.local_rank == 0:
                    raw_uuids = [value["gpu_uuid_raw"] for value in mappings if value["node_rank"] == context.node_rank]
                    sampler = NodeNvmlProcessSampler(
                        raw_uuids,
                        root / f"measured_{current_index:02d}_node_{context.node_rank}.jsonl",
                        node_identity=context.host,
                        execution_attempt=current_index + 1,
                        slurm_allocation_identity=allocation_identity,
                    )
                    sampler.start()
            dist.barrier()
            hook = _CalibrationHook(context.device) if measured else None
            torch.cuda.synchronize(context.device)
            started = time.monotonic_ns()
            try:
                result = train_client(
                    model,
                    dataset,
                    client_id,
                    1,
                    config,
                    context.device,
                    training_seed,
                    payload,
                    measurement_hook=hook,
                )
                torch.cuda.synchronize(context.device)
                dist.barrier()
                duration = (time.monotonic_ns() - started) / 1_000_000_000
            finally:
                if sampler is not None:
                    sampler.stop()
            status = [None]
            if measured and context.local_rank == 0:
                status[0] = {
                    "sample_count": sampler.sample_count // context.devices_per_node,
                    "gpu_uuids": node_mapping[context.host],
                    "path": str(sampler.path),
                }
            dist.broadcast_object_list(
                status,
                src=context.node_leader_rank,
                group=node_process_group(context),
                device=context.control_device,
            )
            leaders = [None for _ in range(context.world_size)]
            dist.all_gather_object(leaders, status[0] if context.local_rank == 0 else None)
            if measured and context.is_coordinator:
                files = {
                    mappings[rank]["host"]: value["path"] for rank, value in enumerate(leaders) if value is not None
                }
                merged = root / f"measured_{current_index:02d}_merged.jsonl"
                merge_node_telemetry(
                    files,
                    merged,
                    expected_uuids_by_node=node_mapping,
                    execution_attempt=current_index + 1,
                    slurm_allocation_identity=allocation_identity,
                )
                telemetry_files.append(str(merged))
            return (
                duration,
                result.state_dict,
                0 if not measured else int(status[0]["sample_count"]),
                [] if not measured else list(status[0]["gpu_uuids"]),
            )

        record = calibrate_measurement(
            run_once,
            model,
            args.repetitions,
            float(requirements["maximum_median_runtime_overhead_fraction"]),
            10,
            float(requirements["minimum_interval_coverage_fraction"]),
            context.device,
            expected_gpu_count=context.devices_per_node,
            topology={
                "node_count": context.node_count,
                "device_count": context.physical_device_count,
                "process_count": context.world_size,
                "sampler_topology": requirements["sampler_topology"],
                "sampling_interval_ms": 100,
            },
            execution_commit=git_metadata()["commit"],
        )
        records = [None for _ in range(context.world_size)]
        dist.all_gather_object(records, record)
        if context.is_coordinator:
            artifact = _aggregate(records, config, telemetry_files)
            path = root / "instrumentation_calibration.json"
            atomic_write_json(path, artifact)
            print(path)
            if not artifact["passed"]:
                raise SystemExit(2)
    finally:
        close_process_context()


if __name__ == "__main__":
    main()
