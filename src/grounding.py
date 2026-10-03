import json
import math
import re
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw
from .vqa import answer_image


def locate_object(image_path, target, model, processor):
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Provide a non-empty target.")

    with Image.open(image_path) as source:
        original = source.convert("RGB")

    side = 560
    resized = original.copy()
    resized.thumbnail((side, side), Image.Resampling.LANCZOS)
    ox = (side - resized.width) // 2
    oy = (side - resized.height) // 2
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(resized, (ox, oy))

    # Confirm the processor preserves the coordinate canvas.
    probe = processor.image_processor(images=[canvas], return_tensors="pt")
    grid = probe["image_grid_thw"][0].tolist()
    patch = processor.image_processor.patch_size
    if (int(grid[2] * patch), int(grid[1] * patch)) != (side, side):
        raise ValueError("Processor changed the grounding canvas dimensions.")

    question = (
        f"Locate {target.strip()} in this {side} by {side} pixel image. "
        "Return only a JSON object with key bbox_format and key bbox. "
        'Use bbox_format="xyxy" and bbox=[x_min, y_min, x_max, y_max]. '
        "Use absolute pixel coordinates measured from the top-left corner. "
        "Give one tight bounding box around the visible target. "
        'If the target is absent, return {"bbox_format":"xyxy","bbox":null}.'
    )

    with tempfile.TemporaryDirectory() as temp:
        canvas_path = Path(temp) / "canvas.png"
        canvas.save(canvas_path)
        response = answer_image(
            canvas_path, question, model, processor, max_new_tokens=128
        )

    result = {
        **response,
        "target": target.strip(),
        "raw_output": response["answer"],
        "canvas_size": [side, side],
        "original_size": list(original.size),
        "resized_size": list(resized.size),
        "padding_xy": [ox, oy],
        "bbox_canvas": None,
        "bbox_original": None,
        "visual_review": "pending",
    }
    overlay = original.copy()

    try:
        text = re.sub(r"^```(?:json)?\s*", "", response["answer"].strip())
        text = re.sub(r"\s*```$", "", text)
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object.")
        if parsed.get("bbox_format") != "xyxy" or "bbox" not in parsed:
            raise ValueError("Missing bbox or unsupported bbox format.")

        box = parsed["bbox"]
        if box is None:
            result["status"] = "target_reported_absent"
            return result, overlay

        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Expected four coordinates.")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float))
            or not math.isfinite(v) for v in box
        ):
            raise ValueError("Coordinates must be finite numbers.")

        x1, y1, x2, y2 = map(float, box)
        if not (0 <= x1 < x2 <= side and 0 <= y1 < y2 <= side):
            raise ValueError("Coordinates are out of bounds or unordered.")

        result["bbox_canvas"] = box
        left, top = max(x1, ox), max(y1, oy)
        right = min(x2, ox + resized.width)
        bottom = min(y2, oy + resized.height)
        if right <= left or bottom <= top:
            raise ValueError("Box lies entirely in padding.")

        result["padding_clipped"] = (
            [left, top, right, bottom] != [x1, y1, x2, y2]
        )
        sx, sy = original.width / resized.width, original.height / resized.height
        mapped = [
            (left - ox) * sx, (top - oy) * sy,
            (right - ox) * sx, (bottom - oy) * sy,
        ]
        result["bbox_original"] = mapped
        result["status"] = "valid_coordinates"

        draw_box = [
            min(original.width - 1, round(mapped[0])),
            min(original.height - 1, round(mapped[1])),
            min(original.width - 1, round(mapped[2])),
            min(original.height - 1, round(mapped[3])),
        ]
        ImageDraw.Draw(overlay).rectangle(draw_box, outline="lime", width=3)

    except (ValueError, TypeError) as error:
        result["status"] = "invalid_output"
        result["error"] = str(error)

    return result, overlay
