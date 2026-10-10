import argparse
from pathlib import Path
import torch
parser = argparse.ArgumentParser()
parser.add_argument("path", type=Path)
args = parser.parse_args()
paths = sorted([x for x in args.path.iterdir() if x.suffix.lower() == ".png"])