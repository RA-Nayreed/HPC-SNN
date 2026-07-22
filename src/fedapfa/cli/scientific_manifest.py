"""Shell-facing validation and expansion for federated scientific manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from fedapfa.configuration import (
    load_comparative_allocations,
    load_comparative_evaluation_manifest,
    load_device_capacity_manifest,
    load_distributed_evaluation_manifest,
    load_evaluation_allocations,
    load_evaluation_manifest,
    load_heterogeneity_context_tasks,
    load_heterogeneity_manifest,
    load_published_fedsnn_manifest,
    load_resource_measurement_manifest,
)


def _collection(path: str) -> str:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict) or not isinstance(value.get("collection"), str):
        raise ValueError("manifest collection is missing")
    return value["collection"]


def _tasks(path: str):
    collection = _collection(path)
    if collection == "heterogeneity_evaluation":
        return load_heterogeneity_manifest(path)
    if collection == "published_fedsnn":
        return load_published_fedsnn_manifest(path)
    if collection == "distributed_evaluation":
        return load_distributed_evaluation_manifest(path)
    if collection == "device_capacity_evaluation":
        return load_device_capacity_manifest(path)
    if collection == "resource_measurement":
        return load_resource_measurement_manifest(path)
    if collection in {"scheduling_evaluation", "hierarchical_reduction_evaluation"}:
        return load_evaluation_manifest(path)
    if collection in {"system_scaling_energy_evaluation", "non_iid_energy_evaluation"}:
        return load_comparative_evaluation_manifest(path)
    raise ValueError(f"unsupported scientific collection: {collection}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and expand a federated scientific manifest.")
    parser.add_argument(
        "action",
        choices=(
            "validate",
            "count",
            "task",
            "context-count",
            "context-task",
            "allocation-count",
            "allocation-task",
        ),
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    context_action = args.action.startswith("context-")
    allocation_action = args.action.startswith("allocation-")
    if context_action:
        values = load_heterogeneity_context_tasks(args.manifest)
    elif allocation_action:
        collection = _collection(args.manifest)
        values = (
            load_comparative_allocations(args.manifest)
            if collection in {"system_scaling_energy_evaluation", "non_iid_energy_evaluation"}
            else load_evaluation_allocations(args.manifest)
        )
    else:
        values = _tasks(args.manifest)
    if args.action == "validate":
        print(
            json.dumps(
                {
                    "collection": _collection(args.manifest),
                    "task_count": len(values),
                    "seeds": sorted({value.seed for value in values}),
                    "experiments": list(dict.fromkeys(value.experiment for value in values)),
                },
                sort_keys=True,
            )
        )
    elif args.action in {"count", "context-count", "allocation-count"}:
        print(len(values))
    else:
        if args.index is None or not 0 <= args.index < len(values):
            parser.error(f"--index must be between 0 and {len(values) - 1}")
        value = values[args.index]
        if context_action:
            print("\t".join((str(value.seed), value.experiment, value.source_record["run_directory"])))
        elif allocation_action:
            fields = (
                [str(value.allocation_index), value.dataset, str(value.seed)]
                if hasattr(value, "allocation_index")
                else [value.dataset, str(value.seed)]
            )
            for treatment, task in zip(value.execution_order, value.tasks, strict=True):
                fields.extend((treatment, str(task.config_path)))
            print("\t".join(fields))
        else:
            fields = [
                str(value.config_path),
                str(value.seed),
                value.dataset,
                value.mode,
                value.protocol,
                value.experiment,
                str(value.config["federated"]["participation_fraction"]),
            ]
            if "parallel_execution" in value.config:
                parallel = value.config["parallel_execution"]
                fields.extend(
                    str(item)
                    for item in (
                        parallel["device_count"],
                        parallel["client_processes_per_device"],
                        parallel["process_count"],
                        parallel["control_backend"],
                        parallel["cuda_process_service"],
                        value.config["output_root"],
                    )
                )
                if "evaluation" in value.config:
                    fields.extend(
                        str(item)
                        for item in (
                            parallel["node_count"],
                            parallel["devices_per_node"],
                            value.config["scheduler"]["strategy"],
                            value.config["aggregation_execution"]["topology"],
                            value.config["evaluation"]["comparison_reference"],
                        )
                    )
                elif "comparative_evaluation" in value.config:
                    fields.extend(
                        str(item)
                        for item in (
                            parallel["node_count"],
                            parallel["devices_per_node"],
                            value.config["scheduler"]["strategy"],
                            value.config["aggregation_execution"]["topology"],
                            value.config["comparative_evaluation"]["treatment_id"],
                            value.config["comparative_evaluation"]["comparison_reference"],
                        )
                    )
            print("\t".join(fields))


if __name__ == "__main__":
    main()
