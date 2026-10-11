import argparse
import subprocess
from pathlib import Path
import torchvision.transforms.functional as F
import torchvision.models.detection as detection
from torchvision.utils import draw_bounding_boxes
from PIL import Image
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=Path,required=True)
parser.add_argument("--frames", type=int, default=5, required=False)
parser.add_argument("-s", "--split", action="store_true")
args = parser.parse_args()
res = '"1920x1080"'
# running ffmpeg
if args.split:
    try:
        result = subprocess.run(f"ffmpeg -hwaccel videotoolbox -i {args.file} -s {res} -r {args.frames} frames/output_%03d.png", shell=True, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError() as e:
        print(e.stderr)

# running object classification on one image
image_path = args.file.parent / "frames" / "output_001.png"
img = Image.open(image_path)
img = img.convert("RGB")
print(img.size,img.mode)

weights = detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = detection.fasterrcnn_resnet50_fpn_v2(weights=weights,box_score_thresh=0.9)
model.eval()

img = F.pil_to_tensor(img)
preprocess = weights.transforms()
batch = [preprocess(img)]
prediction = model(batch)[0]
labels = [weights.meta["categories"][i] for i in prediction["labels"]]
box = draw_bounding_boxes(
    img,
    boxes=prediction["boxes"],
    labels=labels,
    colors="white",
    width=2,
    font="bakery.ttf",
    font_size=50
)
img = F.to_pil_image(box.detach())
img.save("out1.png")