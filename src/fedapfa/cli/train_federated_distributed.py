"""Torchrun entry point for single-node distributed FedAvg."""

from __future__ import annotations

import argparse
import copy
import shlex
import sys

import torch.distributed as dist

from fedapfa.configuration import (
    distributed_execution_identity,
    distributed_scientific_identity,
    load_distributed_evaluation_config,
    validate_distributed_evaluation_config,
)
from fedapfa.distributed.process_context import (
    close_process_context,
    initialize_process_context,
    process_resident_memory_bytes,
    verify_identity_consensus,
)
from fedapfa.federated.checkpointing import configuration_identity, state_identity
from fedapfa.federated.workload import prepare_federated_execution_workload
from fedapfa.training.distributed_federated import train_distributed_federated
from fedapfa.utilities.git_metadata import git_metadata
from fedapfa.utilities.run_records import RunAction, initialize_run, plan_run


def _override(config: dict, args: argparse.Namespace) -> dict:
    resolved = copy.deepcopy(config)
    if args.data_root:
        resolved["dataset"]["root"] = args.data_root
    if args.output_root:
        resolved["output_root"] = args.output_root
    if args.seed is not None:
        resolved["seed"] = args.seed
    validate_distributed_evaluation_config(resolved)
    return resolved


def _broadcast_action(context, action: RunAction | None) -> RunAction:
    value = [
        {
            "run_dir": str(action.run_dir),
            "resume_checkpoint": str(action.resume_checkpoint) if action.resume_checkpoint else None,
            "skip_completed": action.skip_completed,
        }
        if context.is_coordinator
        else None
    ]
    dist.broadcast_object_list(value, src=0, device=context.control_device)
    return RunAction(
        run_dir=value[0]["run_dir"],
        resume_checkpoint=value[0]["resume_checkpoint"],
        skip_completed=value[0]["skip_completed"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one single-node distributed FedAvg evaluation.")
    parser.add_argument("config")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--seed", type=int)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume")
    resume_group.add_argument("--resume-auto", action="store_true")
    args = parser.parse_args()
    config = _override(load_distributed_evaluation_config(args.config), args)
    context = initialize_process_context(config["parallel_execution"])
    try:
        command = shlex.join(
            [sys.executable, "-m", "fedapfa.cli.train_federated_distributed", *sys.argv[1:]]
        )
        coordinator_action = (
            plan_run(config, command, args.resume, args.resume_auto) if context.is_coordinator else None
        )
        action = _broadcast_action(context, coordinator_action)
        if action.skip_completed:
            if context.is_coordinator:
                print(f"completed distributed execution already exists; skipping: {action.run_dir}")
            return

        resident_memory_before_workload = process_resident_memory_bytes()
        workload = prepare_federated_execution_workload(config, coordinator=context.is_coordinator)
        bundle = workload.data
        model = workload.model_factory(config)
        if context.is_coordinator:
            initialize_run(
                config,
                {
                    "train": [int(value) for value in bundle.train_indices],
                    "validation": [int(value) for value in bundle.validation_indices],
                },
                command,
                action.resume_checkpoint,
            )
        dist.barrier()
        identity = {
            "configuration_id": configuration_identity(config),
            "scientific_identity": distributed_scientific_identity(config),
            "execution_identity": distributed_execution_identity(config),
            "git": git_metadata(),
            "dataset_identity": bundle.split_artifact.get("dataset_identity"),
            "split_id": bundle.split_artifact["split_id"],
            "partition_id": bundle.partition.partition_id,
            "model_initialization_id": state_identity(model.state_dict()),
            "resolved_seeds": bundle.resolved_seed_values,
            "world_size": context.world_size,
            "visible_device_count": context.visible_device_count,
        }
        process_records = verify_identity_consensus(
            context,
            identity,
            process_resident_memory_before_workload_bytes=resident_memory_before_workload,
        )
        result = train_distributed_federated(
            model,
            bundle,
            config,
            action.run_dir,
            context,
            process_records,
            action.resume_checkpoint,
            client_training=workload.client_training,
        )
        if context.is_coordinator and not result["completed"]:
            raise SystemExit(2)
    finally:
        close_process_context()


if __name__ == "__main__":
    main()
