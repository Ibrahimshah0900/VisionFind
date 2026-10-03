"""Isolated detector adapter; preserves existing person/face implementations."""
from src.config import MODEL_CACHE_ROOT
from pathlib import Path
import math
import re

SUPPORTED = ('bed', 'bowl', 'dog', 'truck', 'dining table', 'cup')
ALIASES = {name: name for name in SUPPORTED}
ALIASES.update({name + 's': name for name in SUPPORTED})
_model = None
_weights = None


def resolve_target(target):
    text = re.sub(r'\s+', ' ', str(target).lower()).strip(' .?!')
    text = re.sub(r'^(?:(?:all|the|visible|each|every) )+', '', text)
    text = re.sub(r'^instances of ', '', text)
    text = re.sub(r' in (?:the )?(?:image|photo|picture)$', '', text)
    return ALIASES.get(text)


def detect_category(image_path, target):
    global _model, _weights
    category = resolve_target(target)
    if category is None:
        raise ValueError('Unsupported or qualified target; supported: ' + ', '.join(SUPPORTED))
    import torch
    from PIL import Image, ImageDraw, ImageOps
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_320_fpn,
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    )
    if _model is None:
        cache = (MODEL_CACHE_ROOT / 'torch')
        cache.mkdir(parents=True, exist_ok=True)
        previous = torch.hub.get_dir()
        try:
            torch.hub.set_dir(str(cache))
            _weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            _model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=_weights).cpu().eval()
        finally:
            torch.hub.set_dir(previous)
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    tensor = _weights.transforms()(image)
    with torch.inference_mode():
        result = _model([tensor])[0]
    class_id = _weights.meta['categories'].index(category)
    detections = []
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for box, label, score in zip(result['boxes'].tolist(), result['labels'].tolist(), result['scores'].tolist()):
        if label != class_id or score < .55:
            continue
        x1, y1, x2, y2 = box
        if not all(math.isfinite(v) for v in box) or not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            continue
        detections.append({'label': category, 'bbox_original': box, 'score': score})
        draw.rectangle(box, outline='lime', width=3)
        draw.text((x1+3, y1+3), f'{category} {len(detections)}', fill='lime', stroke_width=1, stroke_fill='black')
    return {'detector': 'fasterrcnn_mobilenet_v3_large_320_fpn', 'kind': category,
            'threshold': .55, 'configuration': {'min_size': 320, 'max_size': 640},
            'image_size': list(image.size), 'detected_count': len(detections),
            'detections': detections, 'review_status': 'pending'}, overlay


def chat_reply(engine, image_path, message, plan, folder):
    result, overlay = detect_category(image_path, plan['target'])
    count = result['detected_count']
    category = result['kind']
    # CONCISE_IMAGE_REPLY_V1
    parts = [
        f"I detected **{count} {category} region{'s' if count != 1 else ''}**."
        if count else
        f"I couldn't reliably locate any {category} objects in this image.",
        "Detections are approximate."
    ]
    media = []
    if plan.get('locate'):
        path = Path(folder) / 'general_detections.png'
        overlay.save(path)
        pass
        media.append((str(path), f'Predicted {category} locations'))
    details = {'detection': result, 'count_source': 'dedicated_detector', 'box_count': count}
    if re.search(r'\b(describe|explain|exolain|descirbe)\b', message, re.I):
        try:
            response = engine.ask(image_path, 'Describe the visible scene briefly. Do not count objects or output coordinates. Do not invent details.', max_new_tokens=110)
            details['description'] = response
            sentences = re.split(r'(?<=[.!?])\s+|\n+', response['answer'].strip())
            safe = [s for s in sentences if not re.search(r'\b(?:\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|boxes|coordinates)\b', s, re.I)]
            details['displayed_description'] = ' '.join(safe)
            if safe:
                parts.append('**Scene description:** ' + ' '.join(safe))
        except Exception as error:
            details['description_error'] = f'{type(error).__name__}: {error}'
    return '\n\n'.join(parts), media, details
