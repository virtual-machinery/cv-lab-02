import argparse
import subprocess
from pathlib import Path
import torch
parser = argparse.ArgumentParser()
parser.add_argument("file", type=Path)
parser.add_argument("frames", type=int, default=5)
args = parser.parse_args()
# running ffmpeg
try:
    result = subprocess.run(f"ffmpeg -hwaccel videotoolbox -i {args.file} -r 23 frames/output_%03d.png", shell=True, text=True, capture_output=True)
except subprocess.CalledProcessError():
    print(result.stderr)
