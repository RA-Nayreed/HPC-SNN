"""Create deterministic system-scaling figures from a summary artifact."""

from __future__ import annotations

import argparse

from fedapfa.analysis.comparative_figures import generate_system_scaling_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Create system-scaling/energy figures.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    for path in generate_system_scaling_figures(args.summary, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
