"""Shell-friendly access to the strict federated manifest expansion."""

from __future__ import annotations

import argparse
import json

from fedapfa.configuration import load_federated_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and expand the SHD FedAvg scientific manifest.")
    parser.add_argument("action", choices=("validate", "count", "task"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    tasks = load_federated_manifest(args.manifest)
    if args.action == "validate":
        print(
            json.dumps(
                {
                    "task_count": len(tasks),
                    "seeds": sorted({task.seed for task in tasks}),
                    "experiments": list(dict.fromkeys(task.experiment for task in tasks)),
                },
                sort_keys=True,
            )
        )
    elif args.action == "count":
        print(len(tasks))
    else:
        if args.index is None or not 0 <= args.index < len(tasks):
            parser.error(f"--index must be between 0 and {len(tasks) - 1}")
        task = tasks[args.index]
        print(
            "\t".join(
                (
                    str(task.config_path),
                    str(task.seed),
                    task.dataset,
                    task.mode,
                    task.protocol,
                    task.experiment,
                    str(task.config["federated"]["participation_fraction"]),
                )
            )
        )


if __name__ == "__main__":
    main()
