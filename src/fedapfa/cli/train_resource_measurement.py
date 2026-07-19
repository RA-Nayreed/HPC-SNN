"""Resource-measured execution through the established distributed trainer."""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

from fedapfa.configuration import (
    load_resource_measurement_config,
    validate_resource_measurement_config,
)
from fedapfa.measurement.runtime import ResourceMeasurementSession

from .train_federated_distributed import _override, execute_distributed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one accepted client-resource execution.")
    parser.add_argument("config")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--seed", required=True, type=int)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume")
    resume_group.add_argument("--resume-auto", action="store_true")
    args = parser.parse_args()
    calibration = Path(args.calibration).resolve()
    if not calibration.is_file():
        parser.error("--calibration must name a passing artifact")
    config = _override(
        load_resource_measurement_config(args.config),
        args,
        validate_resource_measurement_config,
    )

    def session_factory(config, run_dir, bundle, model, context):
        return ResourceMeasurementSession(
            config,
            run_dir,
            bundle,
            model,
            context,
            calibration,
        )

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
            "fedapfa.cli.train_resource_measurement",
            session_factory,
        )
        if run_dir is not None:
            acceptance_path = Path(run_dir) / "measurement_acceptance.json"
            if not acceptance_path.is_file() or not json.loads(
                acceptance_path.read_text(encoding="utf-8")
            ).get("accepted"):
                raise RuntimeError(
                    "completed training record lacks accepted resource measurement"
                )
            print(acceptance_path)
            print(Path(run_dir) / "client_resource_records.jsonl")
            print(Path(run_dir) / "device_samples.jsonl")
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
