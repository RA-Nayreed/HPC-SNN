"""Explicit post-selection evaluation of a centralized checkpoint."""

import argparse
import json

from fedapfa.configuration import load_config
from fedapfa.models.model_factory import make_model
from fedapfa.training.centralized import make_loader, resolve_device, run_epoch
from fedapfa.training.checkpointing import load_checkpoint
from fedapfa.training.protocols import prepare_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a selected centralized checkpoint once.")
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    device = resolve_device(config["device"])
    model = make_model(config).to(device)
    load_checkpoint(args.checkpoint, model)
    bundle = prepare_datasets(config)
    dataset = getattr(bundle, args.split)
    if callable(dataset):
        dataset = dataset()
    if dataset is None:
        raise RuntimeError(f"configuration does not expose a {args.split} dataset")
    metrics = run_epoch(
        model,
        make_loader(dataset, config, False),
        device,
        None,
        config["training"]["max_test_batches"]
        if args.split == "test"
        else config["training"]["max_validation_batches"],
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
