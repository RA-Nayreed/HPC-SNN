"""Command-line scheduling-evaluation summary."""

import argparse

from fedapfa.analysis import summarize_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize scheduling evaluation evidence.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument(
        "--data-root",
        required=True,
        help="Absolute runtime data root containing the shd and ssc directories.",
    )
    parser.add_argument("--slurm-accounting", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = summarize_evaluation(
        args.manifest,
        args.runs_root,
        args.output_dir,
        slurm_accounting=args.slurm_accounting,
        data_root=args.data_root,
    )
    if summary["collection"] != "scheduling_evaluation" or not summary["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
