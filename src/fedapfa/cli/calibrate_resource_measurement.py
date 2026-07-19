"""Calibrate resource measurement with one SHD training client."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from fedapfa.configuration import load_resource_measurement_config
from fedapfa.federated.client import train_client
from fedapfa.federated.communication_accounting import model_payload_bytes
from fedapfa.federated.randomness import derive_seed
from fedapfa.federated.workload import prepare_federated_execution_workload
from fedapfa.measurement.calibration import calibrate_measurement
from fedapfa.measurement.clock import CudaTimingAdapter
from fedapfa.measurement.power import NvmlProcessSampler
from fedapfa.measurement.records import read_jsonl
from fedapfa.utilities.serialization import atomic_write_json


class _CalibrationHook:
    def __init__(self, device: torch.device) -> None:
        self.timing = CudaTimingAdapter(device)
        self.result = None
        self.open = False

    def start(self) -> None:
        self.timing.start()
        self.open = True

    def begin_device_work(self):
        return self.timing.begin_device_work()

    def end_device_work(self, token) -> None:
        self.timing.end_device_work(token)

    def observe_batch(self, _batch, _rates) -> None:
        return None

    def finish(self, _data_wait_seconds: float) -> None:
        self.result = self.timing.finish()
        self.open = False

    def abort_if_open(self) -> None:
        self.open = False


def _gpu_uuid(device: torch.device) -> str:
    import os

    declared = os.environ.get("FEDAPFA_GPU_UUID")
    if declared:
        return declared
    value = getattr(torch.cuda.get_device_properties(device), "uuid", None)
    if value is None:
        raise RuntimeError("CUDA device UUID is unavailable")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value)
    return value if value.startswith("GPU-") else f"GPU-{value}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate 100 ms NVML client-resource measurement.")
    parser.add_argument("config")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    root = Path(args.result_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifact = root / "instrumentation_calibration.json"
    samples_path = root / "calibration_device_samples.jsonl"
    if artifact.exists() or samples_path.exists():
        raise FileExistsError("calibration result root already contains finalized records")
    config = load_resource_measurement_config(args.config)
    if config["dataset"]["name"] != "shd":
        parser.error("calibration requires the SHD resource configuration")
    config["dataset"]["root"] = str(Path(args.data_root).resolve())
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("calibration requires exactly one visible CUDA device")
    device = torch.device("cuda", 0)
    workload = prepare_federated_execution_workload(config, coordinator=True)
    model = workload.model_factory(config).to(device)
    bundle = workload.data
    client_id = sorted(bundle.client_ids)[0]
    dataset = bundle.client_dataset(client_id)
    training_seed = derive_seed(
        config["seed"], config["seed_streams"]["client_training"], 1, client_id
    )
    payload = model_payload_bytes(model.state_dict())
    uuid = _gpu_uuid(device)
    samples_path.touch(exist_ok=True)
    sampling_errors = []

    def run_once(measured: bool):
        before = len(read_jsonl(samples_path))
        sampler = None
        hook = None
        if measured:
            sampler = NvmlProcessSampler(
                uuid,
                1,
                samples_path,
                int(config["resource_measurement"]["sampling_interval_ms"]),
            )
            sampler.start()
            hook = _CalibrationHook(device)
        torch.cuda.synchronize(device)
        started = time.monotonic_ns()
        try:
            result = train_client(
                model,
                dataset,
                client_id,
                1,
                config,
                device,
                training_seed,
                payload,
                measurement_hook=hook,
            )
            torch.cuda.synchronize(device)
            duration = (time.monotonic_ns() - started) / 1_000_000_000
        finally:
            if sampler is not None:
                sampler.stop()
        after_records = read_jsonl(samples_path)[before:]
        sampling_errors.extend(
            value["sampling_error_status"]
            for value in after_records
            if value["sampling_error_status"] is not None
        )
        return duration, result.state_dict, len(after_records), sorted({value["gpu_uuid"] for value in after_records})

    rules = config["resource_measurement"]["calibration"]
    record = calibrate_measurement(
        run_once,
        model,
        args.repetitions,
        float(rules["maximum_median_overhead_fraction"]),
        int(rules["minimum_samples_per_client"]),
        float(rules["minimum_client_fraction"]),
        device,
    )
    record["sampling_interval_ms"] = int(config["resource_measurement"]["sampling_interval_ms"])
    record["dataset"] = "shd"
    record["client_id"] = client_id
    record["training_data_only"] = True
    record["sample_record"] = str(samples_path)
    record["sampling_errors"] = sampling_errors
    if sampling_errors:
        record["passed"] = False
        record["validation_findings"].append("sampling_errors_observed")
    atomic_write_json(artifact, record)
    print(artifact)
    print(samples_path)
    if not record["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
