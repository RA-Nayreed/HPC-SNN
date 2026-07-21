import argparse
import json

from fedapfa.configuration import (
    expand_sweep,
    experiment_id,
    load_config,
    load_resolved_config,
    validate_distributed_evaluation_config,
    validate_evaluation_config,
    validate_federated_config,
    validate_resource_measurement_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve and validate an experiment configuration.")
    parser.add_argument("config")
    args = parser.parse_args()
    candidate = load_resolved_config(args.config)
    if candidate.get("execution") == "federated":
        if "resource_measurement" in candidate:
            validate_resource_measurement_config(candidate)
        elif "evaluation" in candidate:
            validate_evaluation_config(candidate)
        elif "parallel_execution" in candidate:
            validate_distributed_evaluation_config(candidate)
        else:
            validate_federated_config(candidate)
        config = candidate
        expanded = [config]
    else:
        config = load_config(args.config)
        expanded = expand_sweep(config)
    print(
        json.dumps(
            {
                "experiment_id": experiment_id(config),
                "expanded_runs": [experiment_id(item) for item in expanded],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
