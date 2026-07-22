"""Record launcher-observed monotonic allocation and sequential-treatment boundaries."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from fedapfa.utilities.serialization import atomic_write_json


def _now() -> dict:
    return {
        "monotonic_timestamp_ns": time.monotonic_ns(),
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("allocation timing record is malformed")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Record comparative allocation timing boundaries.")
    parser.add_argument("action", choices=("start", "begin-treatment", "end-treatment", "complete"))
    parser.add_argument("--path", required=True)
    parser.add_argument("--allocation-index", required=True, type=int)
    parser.add_argument("--collection")
    parser.add_argument("--dataset")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--order")
    parser.add_argument("--treatment")
    parser.add_argument("--position", type=int)
    args = parser.parse_args()
    path = Path(args.path).resolve()
    if args.action == "start":
        if path.exists():
            raise FileExistsError(path)
        if not all(value is not None for value in (args.collection, args.dataset, args.seed, args.order)):
            parser.error("start requires collection, dataset, seed, and order")
        order = args.order.split(",")
        if not order or any(not value for value in order):
            parser.error("execution order must be a nonempty comma-separated list")
        value = {
            "schema_version": 1,
            "allocation_index": args.allocation_index,
            "collection": args.collection,
            "dataset": args.dataset,
            "seed": args.seed,
            "execution_order": order,
            "display_array_task_id": (
                f"{os.environ.get('SLURM_ARRAY_JOB_ID')}_{os.environ.get('SLURM_ARRAY_TASK_ID')}"
            ),
            "raw_slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "launcher_observed_allocation_start": _now(),
            "launcher_observed_allocation_end": None,
            "treatments": [],
            "open_treatment": None,
            "completed": False,
        }
    else:
        value = _load(path)
        if value["allocation_index"] != args.allocation_index or value["completed"]:
            raise ValueError("allocation timing identity or completion state differs")
        if args.action == "begin-treatment":
            if args.treatment is None or args.position is None or value["open_treatment"] is not None:
                parser.error("begin-treatment requires one closed treatment identity")
            expected_position = len(value["treatments"]) + 1
            if args.position != expected_position or value["execution_order"][args.position - 1] != args.treatment:
                raise ValueError("treatment position differs from the declared execution order")
            value["open_treatment"] = {
                "treatment_id": args.treatment,
                "treatment_position": args.position,
                "start": _now(),
                "end": None,
                "launcher_observed_duration_seconds": None,
            }
        elif args.action == "end-treatment":
            current = value["open_treatment"]
            if (
                current is None
                or current["treatment_id"] != args.treatment
                or current["treatment_position"] != args.position
            ):
                raise ValueError("open treatment identity differs")
            current["end"] = _now()
            current["launcher_observed_duration_seconds"] = (
                current["end"]["monotonic_timestamp_ns"] - current["start"]["monotonic_timestamp_ns"]
            ) / 1_000_000_000
            value["treatments"].append(current)
            value["open_treatment"] = None
        else:
            if (
                value["open_treatment"] is not None
                or [item["treatment_id"] for item in value["treatments"]] != value["execution_order"]
            ):
                raise ValueError("allocation cannot complete before every ordered treatment closes")
            value["launcher_observed_allocation_end"] = _now()
            value["launcher_observed_allocation_duration_seconds"] = (
                value["launcher_observed_allocation_end"]["monotonic_timestamp_ns"]
                - value["launcher_observed_allocation_start"]["monotonic_timestamp_ns"]
            ) / 1_000_000_000
            value["completed"] = True
    atomic_write_json(path, value)


if __name__ == "__main__":
    main()
