import argparse

from fedapfa.datasets.download import download_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate official SHD, SSC, or CIFAR-10 files.")
    parser.add_argument("dataset", choices=("shd", "ssc", "cifar10"))
    parser.add_argument("--root", default="data/raw")
    args = parser.parse_args()
    for path in download_dataset(args.dataset, args.root):
        print(path)


if __name__ == "__main__":
    main()
