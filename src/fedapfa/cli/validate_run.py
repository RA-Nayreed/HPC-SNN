import argparse

from fedapfa.configuration import load_config
from fedapfa.configuration.experiment_id import experiment_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an HPC-SNN experiment configuration.")
    parser.add_argument("config")
    args = parser.parse_args()
    print(experiment_id(load_config(args.config)))


if __name__ == "__main__":
    main()
