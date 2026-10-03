"""Reusable pretrained COCO detector candidate; not the chat default."""
from src.config import MODEL_CACHE_ROOT
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

_model = None
_device = None
WEIGHTS = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
THRESHOLD = .55

def supported_categories():
    return [name for name in WEIGHTS.meta['categories'] if name not in {'__background__','N/A'}]

def load():
    global _model, _device
    if _model is None:
        _device = 'cpu'
        if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= 3*1024**3:
            _device = 'cuda'
        previous = torch.hub.get_dir()
        try:
            torch.hub.set_dir(str(MODEL_CACHE_ROOT / 'torch'))
            _model = fasterrcnn_resnet50_fpn_v2(weights=WEIGHTS).to(_device).eval()
        finally:
            torch.hub.set_dir(previous)
    return _model, _device

@torch.inference_mode()
def detect(path, targets=None):
    categories = WEIGHTS.meta['categories']
    supported = supported_categories()
    targets = supported if targets is None else list(dict.fromkeys(targets))
    if not targets or any(name not in supported for name in targets):
        raise ValueError('Name supported COCO categories; use supported_categories().')
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    width, height = image.size
    model, device = load()
    result = model([WEIGHTS.transforms()(image).to(device)])[0]
    detections = []
    selected = {categories.index(name) for name in targets}
    for box, label, score in zip(result['boxes'].cpu().tolist(),result['labels'].cpu().tolist(),result['scores'].cpu().tolist()):
        if label not in selected or score < THRESHOLD:
            continue
        box = [max(0.,min(width-1.,box[0])),max(0.,min(height-1.,box[1])),
               max(0.,min(width-1.,box[2])),max(0.,min(height-1.,box[3]))]
        if np.isfinite(box).all() and box[2] > box[0] and box[3] > box[1]:
            detections.append({'label':categories[label],'bbox_original':box,'score':score})
    return {'detected_count':len(detections),'detections':detections,'image_size':[width,height],
            'targets':targets,'threshold':THRESHOLD,'device':device,'accuracy':'review pending'}

def release():
    global _model, _device
    _model = None
    _device = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
