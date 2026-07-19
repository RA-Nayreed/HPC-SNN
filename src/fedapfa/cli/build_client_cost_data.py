"""Build the accepted client-cost analysis table."""

from __future__ import annotations

import argparse
from pathlib import Path

from fedapfa.cost_estimation.dataset import build_client_cost_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exactly 6,000 accepted client-cost rows.")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()
    runs_root = Path(args.runs_root).resolve()
    result_root = Path(args.result_root).resolve()
    if not runs_root.is_dir():
        parser.error("--runs-root must be an existing directory")
    run_directories = sorted(path.parent for path in runs_root.rglob("measurement_acceptance.json"))
    if len(run_directories) != 6:
        raise ValueError(f"expected six resource run directories, observed {len(run_directories)}")
    build_client_cost_dataset(run_directories, result_root)
    for name in (
        "client_cost_data.csv",
        "client_cost_schema.json",
        "client_cost_provenance.json",
        "excluded_rows.json",
    ):
        print(result_root / name)


if __name__ == "__main__":
    main()
