import argparse
import json

from fedapfa.datasets.validation import validate_hdf5


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and inspect an event-audio HDF5 file.")
    parser.add_argument("path")
    args = parser.parse_args()
    print(json.dumps(validate_hdf5(args.path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
