from src.config import MODEL_CACHE_ROOT

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from huggingface_hub import hf_hub_download
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)

CACHE = MODEL_CACHE_ROOT
PERSON_THRESHOLD = 0.55
FACE_THRESHOLD = 0.75
_person_model = None
_person_weights = None
_face_model_path = None


def load_person_model():
    global _person_model, _person_weights
    if _person_model is None:
        CACHE.mkdir(parents=True, exist_ok=True)
        previous_hub_dir = torch.hub.get_dir()
        try:
            torch.hub.set_dir(str(CACHE / "torch"))
            _person_weights = (
                FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            )
            _person_model = fasterrcnn_mobilenet_v3_large_320_fpn(
                weights=_person_weights
            ).to("cpu").eval()
            # Increase inference resolution to retain more small-person detail.
            _person_model.transform.min_size = (640,)
            _person_model.transform.max_size = 1024
        finally:
            torch.hub.set_dir(previous_hub_dir)
    return _person_model, _person_weights


def load_face_model_path():
    global _face_model_path
    if _face_model_path is None:
        _face_model_path = hf_hub_download(
            repo_id="opencv/opencv_zoo",
            filename=(
                "models/face_detection_yunet/"
                "face_detection_yunet_2023mar.onnx"
            ),
            cache_dir=str(CACHE / "huggingface"),
        )
    return _face_model_path


@torch.inference_mode()
def detect_objects(image_path, kind):
    if kind not in {"person", "face"}:
        raise ValueError("Supported detector kinds: person, face.")

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    width, height = image.size
    candidates = []

    if kind == "person":
        model, weights = load_person_model()
        tensor = weights.transforms()(image)
        predictions = model([tensor])[0]
        person_id = weights.meta["categories"].index("person")

        for box, label, score in zip(
            predictions["boxes"].tolist(),
            predictions["labels"].tolist(),
            predictions["scores"].tolist(),
        ):
            if label == person_id and score >= PERSON_THRESHOLD:
                candidates.append((box, float(score)))

        model_name = "fasterrcnn_mobilenet_v3_large_320_fpn"
        threshold = PERSON_THRESHOLD
        configuration = {"min_size": 640, "max_size": 1024}

    else:
        # Preserve proportions and map face coordinates back to the original.
        scale = min(1.0, 1280 / max(width, height))
        size = (
            max(1, round(width * scale)),
            max(1, round(height * scale)),
        )
        resized = image.resize(size, Image.Resampling.LANCZOS)
        pixels = cv2.cvtColor(np.asarray(resized), cv2.COLOR_RGB2BGR)

        detector = cv2.FaceDetectorYN.create(
            load_face_model_path(), "", size,
            score_threshold=FACE_THRESHOLD,
            nms_threshold=0.3,
            top_k=5000,
        )
        _, faces = detector.detect(pixels)

        if faces is not None:
            sx, sy = width / size[0], height / size[1]
            for face in faces:
                x, y, w, h = map(float, face[:4])
                candidates.append((
                    [x*sx, y*sy, (x+w)*sx, (y+h)*sy],
                    float(face[-1]),
                ))

        model_name = "opencv_yunet_2023mar"
        threshold = FACE_THRESHOLD
        configuration = {"input_size": list(size), "nms_threshold": 0.3}

    detections = []
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    for box, score in candidates:
        x1, y1, x2, y2 = box
        box = [
            max(0.0, min(width-1.0, x1)),
            max(0.0, min(height-1.0, y1)),
            max(0.0, min(width-1.0, x2)),
            max(0.0, min(height-1.0, y2)),
        ]
        if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            continue

        number = len(detections) + 1
        detections.append({
            "label": kind,
            "bbox_original": box,
            "score": score,
        })
        draw.rectangle(box, outline="lime", width=3)
        draw.text(
            (box[0]+3, box[1]+3),
            f"{kind} {number}",
            fill="lime", stroke_width=1, stroke_fill="black",
        )

    return {
        "detector": model_name,
        "kind": kind,
        "threshold": threshold,
        "configuration": configuration,
        "image_size": [width, height],
        "detected_count": len(detections),
        "detections": detections,
        "review_status": "pending",
    }, overlay
