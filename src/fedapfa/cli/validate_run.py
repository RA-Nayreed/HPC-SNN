import argparse
import json

from fedapfa.configuration import expand_sweep, experiment_id, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve and validate an experiment configuration.")
    parser.add_argument("config")
    args = parser.parse_args()
    config = load_config(args.config)
    print(
        json.dumps(
            {
                "experiment_id": experiment_id(config),
                "expanded_runs": [experiment_id(item) for item in expand_sweep(config)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
