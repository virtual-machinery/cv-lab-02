import argparse
from pathlib import Path
from pipeline import split, gif
from segmentation_script import segment
from detection_script import detect

def parse_args():
    # runs ffmpeg, creates folders if they dont exist yet, returns working directory path
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path,required=True)
    parser.add_argument("--frames", type=int, default=5, required=False)
    parser.add_argument("-s", "--split", action="store_true")
    args = parser.parse_args()

    new_path = args.file.parent / "frames"
    new_path.mkdir(exist_ok=True)
    new_path = args.file.parent / "detection_output"
    new_path.mkdir(exist_ok=True)
    new_path = args.file.parent / "segmentation_output"
    new_path.mkdir(exist_ok=True)
    new_path = args.file.parent / "gifs"
    new_path.mkdir(exist_ok=True)

    return args, args.file.parent

def main():
    args, path = parse_args()
    split(args)
    segment(path)
    gif("segmentation_output",1)
    detect(path)
    gif("detection_output",2)
if __name__ == "__main__":
    main()