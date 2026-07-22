"""Summarize completed system-scaling and energy evidence."""

from __future__ import annotations

import argparse

from fedapfa.analysis.comparative_summary import summarize_comparative_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize all 24 system-scaling/energy executions.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slurm-accounting", required=True)
    args = parser.parse_args()
    summary = summarize_comparative_evaluation(
        args.manifest,
        args.runs_root,
        args.output_dir,
        slurm_accounting=args.slurm_accounting,
    )
    if summary["collection"] != "system_scaling_energy_evaluation":
        parser.error("--manifest must identify system_scaling_energy_evaluation")
    print(f"{args.output_dir}/system_scaling_energy_evaluation_summary.json")


if __name__ == "__main__":
    main()
