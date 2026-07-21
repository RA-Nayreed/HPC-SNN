"""Torchrun entry point for scheduling and hierarchical-reduction evaluations."""

from __future__ import annotations

import argparse

from fedapfa.cli.train_federated_distributed import _override, execute_distributed
from fedapfa.configuration import load_evaluation_config, validate_evaluation_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one validated scheduling or hierarchical-reduction treatment.")
    parser.add_argument("config")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--seed", type=int)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume")
    resume_group.add_argument("--resume-auto", action="store_true")
    args = parser.parse_args()
    config = _override(
        load_evaluation_config(args.config),
        args,
        validator=validate_evaluation_config,
    )
    execute_distributed(config, args, "fedapfa.cli.train_system_evaluation")


if __name__ == "__main__":
    main()
