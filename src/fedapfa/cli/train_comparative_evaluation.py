"""Measured distributed entry point for scaling and non-IID evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
from pathlib import Path

from fedapfa.configuration import (
    load_comparative_evaluation_config,
    validate_comparative_evaluation_config,
    validate_resolved_comparative_manifest,
)
from fedapfa.distributed.process_context import canonical_gpu_uuid
from fedapfa.measurement.comparative_runtime import ComparativeMeasurementSession

from .train_federated_distributed import _override, execute_distributed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one measured scaling/energy or non-IID/energy treatment.")
    parser.add_argument("config")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--seed", required=True, type=int)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume")
    resume_group.add_argument("--resume-auto", action="store_true")
    args = parser.parse_args()
    calibration = Path(args.calibration).resolve()
    if not calibration.is_file():
        parser.error("--calibration must name a passing topology-compatible artifact")
    config = _override(
        load_comparative_evaluation_config(args.config),
        args,
        validate_comparative_evaluation_config,
    )
    pair_records = validate_resolved_comparative_manifest(
        args.manifest, data_root=args.data_root, output_root=args.output_root
    )
    current_pair = (
        config["dataset"]["name"],
        int(config["seed"]),
        config["comparative_evaluation"]["treatment_id"],
    )
    if current_pair not in {(value["dataset"], int(value["seed"]), value["treatment_id"]) for value in pair_records}:
        parser.error("resolved treatment is absent from the supplied comparative manifest")
    calibration_bytes = calibration.read_bytes()
    calibration_artifact = json.loads(calibration_bytes)
    config["instrumentation_calibration_identity"] = {
        "sha256": hashlib.sha256(calibration_bytes).hexdigest(),
        "schema_version": calibration_artifact.get("schema_version"),
        "node_count": calibration_artifact.get("node_count"),
        "device_count": calibration_artifact.get("device_count"),
        "process_count": calibration_artifact.get("process_count"),
        "sampler_topology": calibration_artifact.get("sampler_topology"),
        "sampling_interval_ms": calibration_artifact.get("sampling_interval_ms"),
        "gpu_uuids": sorted(canonical_gpu_uuid(value) for value in calibration_artifact.get("gpu_uuids", [])),
        "execution_commit": calibration_artifact.get("execution_commit"),
    }

    def session_factory(config, run_dir, bundle, model, context):
        return ComparativeMeasurementSession(config, run_dir, bundle, model, context, calibration)

    previous_handlers = {}

    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, terminate)
    try:
        run_dir = execute_distributed(
            config,
            args,
            "fedapfa.cli.train_comparative_evaluation",
            session_factory,
        )
        if run_dir is not None:
            acceptance_path = Path(run_dir) / "measurement_acceptance.json"
            if not acceptance_path.is_file() or not json.loads(acceptance_path.read_text(encoding="utf-8")).get(
                "accepted"
            ):
                raise RuntimeError("completed comparative execution lacks accepted energy evidence")
            print(acceptance_path)
            print(Path(run_dir) / "energy_summary.json")
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
