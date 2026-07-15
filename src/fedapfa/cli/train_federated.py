"""Command-line entry point for one single-GPU SHD FedAvg execution."""

from __future__ import annotations

import argparse
import copy
import shlex
import sys

from fedapfa.configuration import load_federated_config, validate_federated_config
from fedapfa.federated.data_protocol import prepare_federated_shd
from fedapfa.training.centralized import resolve_device
from fedapfa.training.federated import make_initialized_federated_model, train_federated
from fedapfa.utilities.run_records import initialize_run, plan_run


def _override(config: dict, args: argparse.Namespace) -> dict:
    resolved = copy.deepcopy(config)
    if args.data_root:
        resolved["dataset"]["root"] = args.data_root
    if args.output_root:
        resolved["output_root"] = args.output_root
    if args.device:
        resolved["device"] = args.device
    if args.seed is not None:
        resolved["seed"] = args.seed
    validate_federated_config(resolved)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one single-GPU SHD FedAvg scientific evaluation.")
    parser.add_argument("config")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume")
    resume_group.add_argument("--resume-auto", action="store_true")
    args = parser.parse_args()
    config = _override(load_federated_config(args.config), args)
    command = shlex.join([sys.executable, "-m", "fedapfa.cli.train_federated", *sys.argv[1:]])
    action = plan_run(config, command, args.resume, args.resume_auto)
    if action.skip_completed:
        print(f"completed federated execution already exists; skipping: {action.run_dir}")
        return
    resolve_device(config["device"])
    bundle = prepare_federated_shd(config)
    model = make_initialized_federated_model(config)
    run_dir = initialize_run(
        config,
        {
            "train": [int(value) for value in bundle.train_indices],
            "validation": [int(value) for value in bundle.validation_indices],
        },
        command,
        action.resume_checkpoint,
    )
    result = train_federated(model, bundle, config, run_dir, action.resume_checkpoint)
    if not result["completed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
