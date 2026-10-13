import argparse
import subprocess
from pathlib import Path
import torchvision.transforms.functional as F
import torchvision.models.detection as detection
from torchvision.utils import draw_bounding_boxes
import torch
from PIL import Image
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=Path,required=True)
parser.add_argument("--frames", type=int, default=5, required=False)
parser.add_argument("-s", "--split", action="store_true")
args = parser.parse_args()
res = "1920x1080"
# running ffmpeg
if args.split:
    try:
        result = subprocess.run(f"ffmpeg -hwaccel videotoolbox -i {args.file} -s {res} -r {args.frames} frames/input_%03d.png", shell=True, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError() as e:
        print(e.stderr)

# model init
weights = detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = detection.fasterrcnn_resnet50_fpn_v2(weights=weights,box_score_thresh=0.9)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()
preprocess = weights.transforms()

# running object detection on one image
image_folder_path = args.file.parent / "frames"
new_path = args.file.parent / "output"
new_path.mkdir(exist_ok=True)

for image_path in sorted(image_folder_path.glob("input_*.png")):
    image = Image.open(image_path)
    image = image.convert("RGB")
    print(f"got image! info: {image_path, image.size,image.mode}. converting to tensor...")
    image = F.pil_to_tensor(image)
    preprocess = weights.transforms()
    batch = [preprocess(image).to(device)]
    with torch.inference_mode():
        prediction = model(batch)[0]
    prediction = {k: v.cpu() for k, v in prediction.items()}
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(
        image,
        boxes=prediction["boxes"],
        labels=labels,
        colors="white",
        width=2,
        font="bakery.ttf",
        font_size=50
    )
    image = F.to_pil_image(box.detach())
    image.save(f"output/out{image_path.stem[-3:]}.png")
config = "scale=-2:480"
try:
    result = subprocess.run(f"ffmpeg -framerate 5 -i out%003d.png -vf {config} out.gif", shell=True,check=True,text=True, capture_output=True)
except subprocess.CalledProcessError as e:
    print(e.stderr)
]