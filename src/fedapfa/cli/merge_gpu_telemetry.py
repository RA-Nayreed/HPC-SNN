"""Validate and merge per-node hierarchical GPU telemetry."""

from __future__ import annotations

import argparse

from fedapfa.measurement.gpu_telemetry import merge_hierarchical_gpu_telemetry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-file", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allocated-gpu-uuids", required=True)
    args = parser.parse_args()
    merge_hierarchical_gpu_telemetry(
        args.node_file,
        args.output,
        args.allocated_gpu_uuids,
    )


if __name__ == "__main__":
    main()
