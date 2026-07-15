"""Command-line entry point for centralized runs and bounded lambda sweeps."""

from __future__ import annotations

import argparse
import copy
import shlex
import sys

from fedapfa.configuration import expand_sweep, load_config, validate_config
from fedapfa.models.model_factory import make_model
from fedapfa.training.centralized import resolve_device, train_centralized
from fedapfa.training.protocols import prepare_datasets
from fedapfa.utilities.run_records import initialize_run


def _override(config, args):
    resolved = copy.deepcopy(config)
    if args.data_root:
        resolved["dataset"]["root"] = args.data_root
    if args.output_root:
        resolved["output_root"] = args.output_root
    if args.device:
        resolved["device"] = args.device
    validate_config(resolved)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one centralized event-audio SNN.")
    parser.add_argument("config")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--resume")
    args = parser.parse_args()
    base = _override(load_config(args.config), args)
    runs = expand_sweep(base)
    if args.resume and len(runs) != 1:
        parser.error("--resume cannot be used with a sweep")
    command = shlex.join([sys.executable, "-m", "fedapfa.cli.train_centralized", *sys.argv[1:]])
    all_accepted = True
    for config in runs:
        validate_config(config)
        if config["mode"] == "deferred":
            raise RuntimeError(f"experiment {config['name']} is marked deferred")
        model = make_model(config)
        resolve_device(config["device"])
        bundle = prepare_datasets(config)
        run_dir = initialize_run(config, bundle.selected_indices, command, args.resume)
        result = train_centralized(model, bundle, config, run_dir, args.resume)
        all_accepted = all_accepted and result["accepted"]
    if not all_accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
