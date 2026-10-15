import torchvision.transforms.functional as F
import torchvision.models.segmentation as segmentation
from torchvision.utils import draw_segmentation_masks
import torch
from PIL import Image
def segment(path):        # running obj segmentation
    # model init
    weights = segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    model = segmentation.deeplabv3_resnet50(weights=weights)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    preprocess = weights.transforms()
    image_folder_path = path / "frames"
    for image_path in sorted(image_folder_path.glob("input_*.png")):
        result = segment_core(preprocess, device, model, weights, image_path)
        F.to_pil_image(result).save(f"segmentation_output/out{image_path.stem[-3:]}.png")

def segment_core(preprocess, device, model, weights, image_path):
    image = Image.open(image_path).convert("RGB")
    print(f"got image! info: {image_path, image.size, image.mode}. converting to tensor...")
    image = F.pil_to_tensor(image)
    batch = preprocess(image).to(device).unsqueeze(0)
    print("running inference...")
    with torch.inference_mode():
        out = model(batch)["out"]
    label_map = out.argmax(dim=1)[0].cpu()
    masks = label_map == torch.arange(len(weights.meta["categories"]))[:, None, None]
    image = F.resize(image, list(label_map.shape))
    result = draw_segmentation_masks(
        image=image,
        masks=masks[1:],
        alpha=0.8
    )
    return result


