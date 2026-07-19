"""Validate the complete collection and write summaries and figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.resource_measurement import write_resource_figures
from fedapfa.cost_estimation.artifacts import summarize_resource_measurement


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize six accepted resource measurement runs.")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--slurm-accounting", required=True)
    args = parser.parse_args()
    runs_root = Path(args.runs_root).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    result_root = Path(args.result_root).resolve()
    run_directories = sorted(path.parent for path in runs_root.rglob("measurement_acceptance.json"))
    summary = summarize_resource_measurement(
        run_directories,
        artifact_root,
        result_root,
        Path(args.slurm_accounting).resolve(),
    )
    write_resource_figures(artifact_root, result_root, run_directories)
    for path in sorted(value for value in result_root.rglob("*") if value.is_file()):
        print(path)
    if not summary["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
