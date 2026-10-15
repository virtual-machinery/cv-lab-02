import subprocess

def split(args):
    res = "1920x1080"
    if args.split:
        try:
            result = subprocess.run(f"ffmpeg -hwaccel videotoolbox -i {args.file} -s {res} -r {args.frames} frames/input_%03d.png", shell=True, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(e.stderr)
    return args.file.parent
def gif(folder_name, c):
    config = "scale=-2:480"
    # compiling gif
    try:
        result = subprocess.run(f"ffmpeg -framerate 5 -i {folder_name}/out%003d.png -vf {config} gifs/out{c}.gif", shell=True,
                                check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(e.stderr)
