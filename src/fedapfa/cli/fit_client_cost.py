"""Fit and evaluate client-cost models from accepted data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fedapfa.configuration import load_resource_measurement_config
from fedapfa.cost_estimation.artifacts import fit_client_cost_models, load_cost_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit client-cost candidates with seed-separated evaluation.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()
    data_path = Path(args.data).resolve()
    provenance_path = Path(args.provenance).resolve()
    result_root = Path(args.result_root).resolve()
    if not data_path.is_file() or not provenance_path.is_file():
        parser.error("--data and --provenance must be existing files")
    rows = load_cost_csv(data_path)
    if len(rows) != 6000:
        raise ValueError(f"cost fitting requires 6,000 rows, observed {len(rows)}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    config = load_resource_measurement_config(args.config)
    fit_client_cost_models(rows, result_root, config["cost_estimation"], provenance)
    for name in (
        "cost_model_evaluation.json",
        "cost_model_evaluation.csv",
        "cost_model_evaluation.md",
        "client_cost_model.json",
        "energy_cost_model.json",
        "assignment_readiness.json",
    ):
        print(result_root / name)


if __name__ == "__main__":
    main()
