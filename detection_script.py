import torchvision.transforms.functional as F
import torchvision.models.detection as detection
from torchvision.utils import draw_bounding_boxes
import torch
from PIL import Image

def detect(path):
    weights = detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = detection.fasterrcnn_resnet50_fpn_v2(weights=weights,box_score_thresh=0.9)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    preprocess = weights.transforms()
    image_folder_path = path / "frames"
    for image_path in sorted(image_folder_path.glob("input_*.png")):
        result = detect_core(preprocess, device, model, weights, image_path)
        F.to_pil_image(result).save(f"detection_output/out{image_path.stem[-3:]}.png")

def detect_core(preprocess, device, model, weights, image_path):
    image = Image.open(image_path).convert("RGB")
    print(f"got image! info: {image_path, image.size,image.mode}. converting to tensor...")
    image = F.pil_to_tensor(image)
    batch = [preprocess(image).to(device)]
    with torch.inference_mode():
        prediction = model(batch)[0]
    prediction = {k: v.cpu() for k, v in prediction.items()}
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    result = draw_bounding_boxes(
        image,
        boxes=prediction["boxes"],
        labels=labels,
        colors="navy",
        width=5,
        font="bakery.ttf",
        font_size=50
    )
    return result
