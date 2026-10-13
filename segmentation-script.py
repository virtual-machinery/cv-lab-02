import argparse
import subprocess
from pathlib import Path
import torchvision.transforms.functional as F
import torchvision.models.segmentation as segmentation
from torchvision.utils import draw_segmentation_masks
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
    except subprocess.CalledProcessError as e:
        print(e.stderr)

# model init
weights = segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
model = segmentation.deeplabv3_resnet50(weights=weights)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()
preprocess = weights.transforms()

# running object detection
image_folder_path = args.file.parent / "frames"
new_path = args.file.parent / "output"
new_path.mkdir(exist_ok=True)

for image_path in sorted(image_folder_path.glob("input_*.png")):
    image = Image.open(image_path)
    image = image.convert("RGB")
    print(f"got image! info: {image_path, image.size,image.mode}. converting to tensor...")
    image = F.pil_to_tensor(image)
    preprocess = weights.transforms()
    batch = preprocess(image).to(device).unsqueeze(0)
    with torch.inference_mode():
        out = model(batch)["out"]
    label_map = out.argmax(dim=1)[0]
    categories = weights.meta["categories"]
    dog_mask = (label_map == 12)
    image = F.resize(image, list(label_map.shape))
    result = draw_segmentation_masks(
        image=image,
        masks=dog_mask,
        alpha=0.8,
        colors="red"
    )
    F.to_pil_image(result).save(f"output/out{image_path.stem[-3:]}.png")
config = "scale=-2:480"

#compiling gif
try:
    result = subprocess.run(f"ffmpeg -framerate 5 -i output/out%003d.png -vf {config} out.gif", shell=True,check=True,text=True, capture_output=True)
except subprocess.CalledProcessError as e:
    print(e.stderr)