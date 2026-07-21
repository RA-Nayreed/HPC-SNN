"""Generate deterministic figures from a validated summary artifact."""

import argparse

from fedapfa.analysis.evaluation_figures import generate_evaluation_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scheduling or hierarchical figures.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    generate_evaluation_figures(args.summary, args.output_dir)


if __name__ == "__main__":
    main()
