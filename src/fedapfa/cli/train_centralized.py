"""Command-line entry point for centralized runs and reduced-sample lambda grids."""

from __future__ import annotations

import argparse
import copy
import shlex
import sys

from fedapfa.configuration import expand_sweep, load_config, validate_config
from fedapfa.models.model_factory import make_model
from fedapfa.training.centralized import resolve_device, seed_everything, train_centralized
from fedapfa.training.protocols import prepare_datasets
from fedapfa.utilities.run_records import initialize_run, plan_run


def _override(config, args):
    resolved = copy.deepcopy(config)
    if args.data_root:
        resolved["dataset"]["root"] = args.data_root
    if args.output_root:
        resolved["output_root"] = args.output_root
    if args.device:
        resolved["device"] = args.device
    if getattr(args, "seed", None) is not None:
        resolved["seed"] = args.seed
    validate_config(resolved)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one centralized event-audio SNN.")
    parser.add_argument("config")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume")
    resume_group.add_argument("--resume-auto", action="store_true")
    args = parser.parse_args()
    base = _override(load_config(args.config), args)
    runs = expand_sweep(base)
    if args.resume and len(runs) != 1:
        parser.error("--resume cannot be used with a sweep")
    command = shlex.join([sys.executable, "-m", "fedapfa.cli.train_centralized", *sys.argv[1:]])
    all_completed = True
    for config in runs:
        validate_config(config)
        action = plan_run(config, command, args.resume, args.resume_auto)
        if action.skip_completed:
            print(f"completed run already exists; skipping: {action.run_dir}")
            continue
        if config["dataset"]["name"] == "cifar10":
            seed_everything(config["seed"])
        model = make_model(config)
        resolve_device(config["device"])
        bundle = prepare_datasets(config)
        run_dir = initialize_run(
            config,
            bundle.selected_indices,
            command,
            action.resume_checkpoint,
        )
        result = train_centralized(model, bundle, config, run_dir, action.resume_checkpoint)
        all_completed = all_completed and result["completed"]
    if not all_completed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
