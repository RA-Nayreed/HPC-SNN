import argparse
from fedapfa.datasets.shd import EventAudioDataset
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("path"); args=parser.parse_args(); print(EventAudioDataset(args.path).metadata())
