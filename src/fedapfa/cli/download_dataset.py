import argparse
from fedapfa.datasets.download import download_dataset
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("dataset",choices=["shd","ssc"]); parser.add_argument("--root",default="data/raw"); args=parser.parse_args(); download_dataset(args.dataset,args.root)
