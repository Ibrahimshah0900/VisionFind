from PIL import Image, ImageOps, ImageDraw

import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageOps


def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


@torch.inference_mode()
def route_request(engine, message, history, has_image):
    """Choose only supported operations; model output never executes code."""
    system = """
You route requests for an image assistant. Return ONLY JSON:
{
  "mode": "answer" or "search" or "classify",
  "query": "",
  "labels": [],
  "locate": false,
  "target": ""
}

Rules:
- Use search only for requests to find/retrieve images from a collection.
- Questions and descriptions about the current image use answer.
- Use classify when the user explicitly supplies alternative category labels.
- Set locate=true when the user asks to draw, box, highlight, or locate
  objects in the current image. Include the requested target in target.
- A request to count AND draw boxes uses answer with locate=true.
- Counting alone uses answer with locate=false.
- Resolve follow-up references using conversation context where possible.
- Do not invent classification labels.
- Search query must be a concise image description.
- If a request is outside image analysis/search, use answer.
- No tools beyond these are available.
Examples:
"Find photos of coffee" -> {"mode":"search","query":"coffee","labels":[],"locate":false,"target":""}
"Is this a cat or a dog?" -> {"mode":"classify","query":"","labels":["cat","dog"],"locate":false,"target":""}
"How many people? Draw boxes around them." -> {"mode":"answer","query":"","labels":[],"locate":true,"target":"all visible people"}
"""
    payload = json.dumps({
        "has_current_image": has_image,
        "recent_conversation": history[-6:],
        "request": message,
    })
    tokenizer = engine.vlm_processor.tokenizer
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": payload}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(engine.vlm.device)
    output = engine.vlm.generate(
        **inputs, max_new_tokens=220, do_sample=False
    )
    raw = tokenizer.decode(
        output[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    plan = unwrap_object(parse_json(raw))
    if not isinstance(plan, dict):
        raise ValueError("Router did not return an object.")
    if plan.get("mode") not in {"answer", "search", "classify"}:
        raise ValueError("Unsupported route.")
    if not isinstance(plan.get("locate"), bool):
        raise ValueError("Invalid locate flag.")
    for field in ["query", "target"]:
        if not isinstance(plan.get(field), str):
            raise ValueError(f"Invalid {field}.")
    labels = plan.get("labels")
    if not isinstance(labels, list) or any(
        not isinstance(x, str) or not x.strip() for x in labels
    ):
        raise ValueError("Invalid labels.")
    if plan["mode"] == "classify" and not 2 <= len(labels) <= 30:
        raise ValueError("Classification needs 2–30 labels.")
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate labels.")
    if plan["mode"] == "search" and not plan["query"].strip():
        raise ValueError("Empty search query.")
    if plan["locate"] and not plan["target"].strip():
        raise ValueError("Empty localization target.")
    return plan, raw


def locate_many(engine, image_path, target, folder):
    """Generate boxes on a fixed canvas; validate and map each box."""
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    side = 560
    resized = original.copy()
    resized.thumbnail((side, side), Image.Resampling.LANCZOS)
    ox, oy = (side - resized.width) // 2, (side - resized.height) // 2
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(resized, (ox, oy))
    canvas_path = folder / "grounding_input.png"
    canvas.save(canvas_path)

    processor = engine.vlm_processor.image_processor
    probe = processor(images=[canvas], return_tensors="pt")
    _, gh, gw = probe["image_grid_thw"][0].tolist()
    if (int(gw * processor.patch_size), int(gh * processor.patch_size)) != (side, side):
        raise ValueError("Unexpected grounding preprocessing dimensions.")

    prompt = (
        f"Locate {target} in this 560 by 560 pixel image. "
        "Return only JSON with this schema: "
        '{"boxes":[{"label":"object name","bbox":[x_min,y_min,x_max,y_max]}]}. '
        "Use absolute pixel coordinates from the top-left corner, not 0–1000 "
        "normalized coordinates. Return a separate tight box for each requested "
        "visible object, at most 12 boxes. If absent, return an empty boxes list."
    )
    response = engine.ask(canvas_path, prompt, max_new_tokens=640)
    result = {
        "target": target,
        "raw_output": response["answer"],
        "canvas_size": [side, side],
        "original_size": list(original.size),
        "resized_size": list(resized.size),
        "padding_xy": [ox, oy],
        "valid_boxes": [],
        "rejected_boxes": [],
        "visual_review": "pending",
    }

    try:
        parsed = parse_json(response["answer"])
        boxes = normalize_boxes(parsed, rejected=result["rejected_boxes"])
        if not isinstance(boxes, list) or len(boxes) > 12:
            raise ValueError("Expected a list with at most 12 boxes.")
    except (ValueError, TypeError, KeyError) as error:
        result["error"] = str(error)
        return result, None

    overlay = original.copy()
    draw = ImageDraw.Draw(overlay)
    colors = ["#00ff70", "#ffb000", "#00bfff", "#ff60c0"]

    for i, item in enumerate(boxes):
        try:
            if not isinstance(item, dict):
                raise ValueError("Box entry must be an object.")
            label = item["label"]
            box = item["bbox"]
            if not isinstance(label, str) or not label.strip():
                raise ValueError("Missing label.")
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("Expected four coordinates.")
            if any(
                isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(v) for v in box
            ):
                raise ValueError("Coordinates must be finite numbers.")
            x1, y1, x2, y2 = map(float, box)
            if not (0 <= x1 < x2 <= side and 0 <= y1 < y2 <= side):
                raise ValueError("Out-of-range or unordered coordinates.")

            a, b = max(x1, ox), max(y1, oy)
            c, d = min(x2, ox + resized.width), min(y2, oy + resized.height)
            if c <= a or d <= b:
                raise ValueError("Box falls in padding.")
            sx, sy = original.width / resized.width, original.height / resized.height
            mapped = [(a-ox)*sx, (b-oy)*sy, (c-ox)*sx, (d-oy)*sy]
            result["valid_boxes"].append({
                "label": label,
                "bbox_canvas": box,
                "bbox_original": mapped,
                "padding_clipped": [a,b,c,d] != [x1,y1,x2,y2],
            })
            coordinates = [
                min(original.width-1, round(mapped[0])),
                min(original.height-1, round(mapped[1])),
                min(original.width-1, round(mapped[2])),
                min(original.height-1, round(mapped[3])),
            ]
            color = colors[i % len(colors)]
            draw.rectangle(coordinates, outline=color, width=4)
            draw.text(
                (coordinates[0]+4, coordinates[1]+4),
                f"{i+1}. {label}", fill=color,
                stroke_width=1, stroke_fill="black",
            )
        except (ValueError, TypeError, KeyError) as error:
            result["rejected_boxes"].append({"index": i, "error": str(error)})

    if result["valid_boxes"]:
        path = folder / "annotated.png"
        overlay.save(path)
        return result, str(path)
    return result, None


def build_chat(engine):
    root = Path(engine.root)

    def respond(message, upload, chat, state):
        if not message or not message.strip():
            raise gr.Error("Type what you want to know or do.")
        if len(message) > 2000:
            raise gr.Error("Please keep each message below 2,000 characters.")

        message = message.strip()
        chat = list(chat or [])
        state = dict(state or {})
        history = list(state.get("history", []))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        folder = root / "results/chat" / stamp
        folder.mkdir(parents=True, exist_ok=False)

        # Copy a new upload to durable storage and reset image-specific context.
        if upload:
            with Image.open(upload) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            digest = hashlib.sha256(
                str(image.size).encode() + image.tobytes()
            ).hexdigest()
            if digest != state.get("image_hash"):
                path = folder / "input.png"
                image.save(path)
                state.update(image_path=str(path), image_hash=digest)
                history = []
                chat.append({
                    "role": "user",
                    "content": gr.Image(value=str(path)),
                })

        image_path = state.get("image_path")
        chat.append({"role": "user", "content": message})
        started = time.perf_counter()
        record = {"message": message, "image_path": image_path}
        reply_parts = []
        media = []

        try:
            plan, raw_plan = route_request(
                engine, message, history, bool(image_path)
            )
            record.update(plan=plan, router_output=raw_plan)

            if plan["mode"] == "clarify":
                reply_parts.append(plan["clarification"])

            elif plan["mode"] == "search":
                matches = engine.search(plan["query"], top_k=3)
                record["search_results"] = matches.to_dict(orient="records")
                reply_parts.append(
                    "Here are the closest matches in your indexed collection:"
                )
                for _, row in matches.iterrows():
                    media.append((
                        str(root / row["relative_path"]),
                        f"{int(row['rank'])}. {row['filename']} — "
                        f"cosine similarity {row['similarity']:.3f}",
                    ))

            elif not image_path:
                reply_parts.append(
                    "Upload an image so I can help with that. "
                    "You can also ask me to find images in your saved collection."
                )

            elif detector_kind(plan, message) is not None:
                detector_parts, detector_media, detector_details = detector_reply(
                    engine, image_path, message, plan, folder
                )
                reply_parts.extend(detector_parts)
                media.extend(detector_media)
                record["detector_result"] = detector_details

            else:
                if plan["mode"] == "classify":
                    ranked = engine.classify(image_path, plan["labels"])
                    record["classification"] = ranked.to_dict(orient="records")
                    reply_parts.append("Best matches among your labels:\n\n" + "\n".join(
                        f"- {row['label']}: {row['cosine_similarity']:.3f}"
                        for _, row in ranked.iterrows()
                    ) + "\n\nThese are cosine similarities, not probabilities.")

                elif not plan["locate"] or needs_text_answer(message):
                    context = json.dumps(history[-6:], ensure_ascii=False)
                    prompt = (
                        "You are VisionFind, an image assistant. Answer the "
                        "current request using the attached image. Recent chat "
                        "is context, not verified visual evidence. Do not invent "
                        "details. If asked to draw boxes, answer the text portion "
                        "only; a separate tool will draw them. Do not output "
                        "coordinates or claim that you already drew anything. "
                        "If outside image analysis, explain the supported scope.\n"
                        f"Recent chat: {context}\nCurrent request: {message}"
                    )
                    answer = engine.ask(image_path, prompt, max_new_tokens=220)
                    record["answer"] = answer
                    reply_parts.append(answer["answer"])

                if plan["locate"]:
                    grounding, annotated = locate_many(
                        engine, image_path, plan["target"], folder
                    )
                    record["grounding"] = grounding
                    count = len(grounding["valid_boxes"])
                    if annotated:
                        reply_parts.append(
                            f"I drew {count} predicted box(es). "
                            "They are approximate and may miss or duplicate objects."
                        )
                        media.append((annotated, "Predicted object locations"))
                    elif grounding.get("error") or grounding["rejected_boxes"]:
                        reply_parts.append(
                            "I couldn't produce valid boxes for this request. "
                            "Try describing the target more specifically."
                        )
                    else:
                        reply_parts.append(
                            "The localization model returned no boxes; "
                            "that does not prove the object is absent."
                        )

        except Exception as error:
            # Preserve diagnostic details in Drive; keep the chat understandable.
            record["error"] = f"{type(error).__name__}: {error}"
            reply_parts.append(
                "I couldn't complete that request. Try a simpler description "
                "such as “Describe this image” or “Draw a box around the person.” "
                f"Diagnostic: {type(error).__name__}: {str(error)[:180]}. "
                "Full details were saved with this chat turn."
            )

        reply = "\n\n".join(reply_parts)
        chat.append({"role": "assistant", "content": reply})
        for path, caption in media:
            chat.append({"role": "assistant", "content": caption})
            chat.append({
                "role": "assistant",
                "content": gr.Image(value=path),
            })

        history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply[:2000]},
        ])
        state["history"] = history[-8:]
        record.update(
            reply=reply,
            recent_history=state["history"],
            elapsed_seconds=round(time.perf_counter()-started, 3),
            model_revision=getattr(engine.vlm.config, "_commit_hash", None),
        )
        (folder / "turn.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        return "", chat, state

    css = """
    .gradio-container {max-width: 1080px !important; margin: auto;}
    #hero {padding: 20px 24px; border-radius: 18px;
           background: linear-gradient(120deg,#101c35,#214c62);}
    #hero h1, #hero p {color: white !important;}
    """
    with gr.Blocks(
        title="VisionFind Chat",
        theme=gr.themes.Soft(primary_hue="teal"),
        css=css,
    ) as app:
        gr.HTML(
            '<div id="hero"><h1>VisionFind</h1>'
            '<p>Your images. Your questions. One conversation.</p></div>'
        )
        state = gr.State({})
        with gr.Row():
            with gr.Column(scale=1, min_width=240):
                upload = gr.Image(
                    type="filepath", sources=["upload"],
                    label="Current image", height=290,
                )
                gr.Markdown(
                    "Upload once, then ask follow-up questions.\n\n"
                    "**Try:**\n"
                    "- Describe this photo.\n"
                    "- How many people? Draw boxes around them.\n"
                    "- Now locate the hat.\n"
                    "- Find photos of coffee.\n\n"
                    "Uploads and chat results are saved to Drive. "
                    "Predictions can be wrong. Search uses your existing collection."
                )
            with gr.Column(scale=3):
                chat = gr.Chatbot(
                    type="messages", label="Conversation", height=560,
                )
                message = gr.Textbox(
                    placeholder="Ask about your image or search your collection…",
                    label="Message", lines=2,
                )
                send = gr.Button("Send", variant="primary")

        options = {
            "concurrency_limit": 1,
            "concurrency_id": "visionfind_chat",
            "api_name": False,
        }
        send.click(
            respond, [message, upload, chat, state],
            [message, chat, state], **options,
        )
        message.submit(
            respond, [message, upload, chat, state],
            [message, chat, state], **options,
        )

    return app


# VISIONFIND_CHAT_REPAIR_V1

def unwrap_object(value):
    """Accept an object wrapped in one or more singleton lists."""
    for _ in range(6):
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        else:
            break
    return value


def normalize_boxes(value, rejected=None):
    """Normalize structure only; never rescale or invent coordinates."""
    def collect(node, depth=0):
        if depth > 8:
            raise ValueError("Bounding-box JSON is nested too deeply.")
        if isinstance(node, list):
            output = []
            for child in node:
                output.extend(collect(child, depth + 1))
            return output
        if not isinstance(node, dict):
            raise ValueError("Expected bounding-box objects.")
        if "boxes" in node:
            if not isinstance(node["boxes"], list):
                raise ValueError("'boxes' must be a list.")
            return collect(node["boxes"], depth + 1)

        box = node.get("bbox", node.get("bbox_2d"))
        if box is None:
            if rejected is None:
                raise ValueError("Missing bbox or bbox_2d.")
            rejected.append({"entry": node, "error": "Missing bbox or bbox_2d."})
            return []
        label = node.get("label", "requested object")
        return [{"label": label, "bbox": box}]

    boxes = collect(value)
    if len(boxes) > 12:
        raise ValueError("More than 12 boxes returned.")
    return boxes


def is_explicit_box_request(message):
    return bool(re.search(
        r"\b(draw|show|put|add|make|create)\b.{0,90}\bbox(?:es)?\b"
        r"|\bbounding\s*box(?:es)?\b"
        r"|\b(locate|highlight|outline)\b",
        message,
        flags=re.IGNORECASE,
    ))


def needs_text_answer(message):
    """Keep text answers for mixed requests; skip them for box-only requests."""
    return bool(re.search(
        r"\b(how|what|why|who|which|describe|explain|count|"
        r"tell|rate|rating|pretty|beautiful|wearing|color|colour)\b",
        message,
        flags=re.IGNORECASE,
    ))


_model_route_request = route_request

def route_request(engine, message, history, has_image):
    # Explicit drawing requests should not depend on the model formatting a plan.
    if is_explicit_box_request(message):
        # The visual grounding model interprets the target from the request.
        # Include brief context for follow-ups such as "box around him".
        context = json.dumps(history[-4:], ensure_ascii=False)
        target = (
            "the object(s) requested in the CURRENT REQUEST below. "
            "Use recent conversation only to resolve references; use the image "
            "to determine what is visible. Ignore non-localization requests.\n"
            f"Recent conversation: {context}\nCURRENT REQUEST: {message}"
        )
        plan = {
            "mode": "answer",
            "query": "",
            "labels": [],
            "locate": True,
            "target": target,
        }
        return plan, "Explicit localization request: direct route"

    return _model_route_request(engine, message, history, has_image)


# QUERY_ROUTER_PHASE1
from .query_router import route_request


# DETECTOR_CHAT_INTEGRATION_V1
from .detectors import detect_objects


def detector_kind(plan, message):
    """Use dedicated detectors for broad person/face requests only."""
    if plan.get("mode") != "answer":
        return None

    text = normalize_detector_text(message)
    target = normalize_detector_text(plan.get("target", ""))
    counting = bool(re.search(r"\b(how many|count|number of)\b", text))
    drawing = bool(plan.get("locate"))

    if not (counting or drawing):
        return None

    # Do not silently turn a selected person or subgroup into ALL people.
    restricted = re.search(
        r"\b(left|right|middle|first|second|third|wearing|holding|"
        r"red|blue|green|black|white|shirt|jacket|hat|behind|beside)\b",
        target,
    )
    if restricted:
        return None

    # Restrict detection routing to broad targets.
    broad_face = re.fullmatch(
        r"(?:(?:all|the|visible|human|each|every)\s+)*"
        r"faces?(?:\s+in\s+(?:the\s+)?(?:image|photo|picture))?",
        target,
    )
    broad_person = re.fullmatch(
        r"(?:(?:all|the|visible|each|every)\s+)*"
        r"(?:people|persons?|humans?)"
        r"(?:\s+in\s+(?:the\s+)?(?:image|photo|picture))?",
        target,
    )

    if broad_face:
        return "face"
    if broad_person:
        return "person"

    # Count-only plans sometimes leave target empty.
    if not target and counting:
        if re.search(r"\bfaces?\b", text):
            return "face"
        if re.search(r"\b(people|persons?|ppl)\b", text):
            return "person"
    return None


def normalize_detector_text(text):
    return re.sub(r"\s+", " ", text.lower()).strip(" .?!")


def detector_reply(engine, image_path, message, plan, folder):
    kind = detector_kind(plan, message)
    result, overlay = detect_objects(image_path, kind)
    # IMAGE_PERSON_FACE_COVERAGE_V2
    if kind == "person":
        clipped_faces = []
        if result["detected_count"] > 0:
            try:
                face_result, _ = detect_objects(image_path, "face")
                face_boxes = [
                    item["bbox_original"]
                    for item in face_result.get("detections", [])
                    if item.get("label") == "face"
                ]
                person_boxes = [
                    item["bbox_original"]
                    for item in result["detections"]
                ]
                clipped_faces = _vf_clipped_faces(person_boxes, face_boxes)
            except Exception as error:
                result = dict(result)
                result["face_coverage_check_error"] = str(error)

        if result["detected_count"] == 0 or clipped_faces:
            from src import detector_candidate
            try:
                stronger = detector_candidate.detect(
                    str(image_path), targets=["person"]
                )
            finally:
                detector_candidate.release()

            stronger_boxes = [
                item["bbox_original"]
                for item in stronger["detections"]
            ]
            covers_faces = all(
                any(_vf_face_coverage(box, face) >= 0.9
                    for box in stronger_boxes)
                for face in clipped_faces
            )

            if covers_faces:
                baseline = result
                result = dict(baseline)
                result.update(
                    detector="fasterrcnn_resnet50_fpn_v2",
                    threshold=stronger["threshold"],
                    detected_count=len(stronger["detections"]),
                    detections=stronger["detections"],
                    fallback_from=baseline["detector"],
                    fallback_reason=(
                        "face_outside_person_box"
                        if clipped_faces else "zero_person_detections"
                    ),
                )
                with Image.open(image_path) as original_image:
                    overlay = ImageOps.exif_transpose(
                        original_image
                    ).convert("RGB")
                draw = ImageDraw.Draw(overlay)
                for detection in result["detections"]:
                    draw.rectangle(
                        detection["bbox_original"],
                        outline="lime", width=3,
                    )
            else:
                result = dict(result)
                result["coverage_review"] = "stronger boxes still exclude face"


    count = result["detected_count"]
    noun = "face" if kind == "face" else "person region"
    plural = noun if count == 1 else noun + "s"

    # CONCISE_IMAGE_REPLY_V1
    subject = ("face" if count == 1 else "faces") if kind == "face" else (
        "person" if count == 1 else "people"
    )
    parts = [
        f"I found **{count} {subject}**." if count else
        "I couldn't reliably locate any matching subjects in this image.",
        "Detections are approximate."
    ]
    media = []
    if plan.get("locate"):
        output_path = folder / f"{kind}_detections.png"
        overlay.save(output_path)
        if count:
            pass
        else:
            pass
        media.append((str(output_path), f"Predicted {kind} locations"))

    details = {
        "detection": result,
        "count_source": "dedicated_detector",
        "box_count": count,
        "description": None,
    }

    # For a combined request, describe the scene separately from counting.
    if re.search(
        r"\b(explain|exolain|describe|desribe|discribe|description|"
        r"what is happening|what's happening)\b",
        message, re.IGNORECASE,
    ):
        try:
            description = engine.ask(
                image_path,
                "Describe the visible scene briefly, focusing on actions "
                "and surroundings. Do not count people or faces. Do not "
                "mention numbers, boxes, coordinates, or detection results. "
                "Do not guess the event, location, or time of day.",
                max_new_tokens=110,
            )
            # Avoid displaying explicit numerical count claims in this
            # optional description. Preserve the raw response in the log.
            sentences = re.split(
                r"(?<=[.!?])\s+|\n+", description["answer"].strip()
            )
            count_pattern = (
                r"\b(?:\d+|zero|one|two|three|four|five|six|seven|eight|"
                r"nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
                r"sixteen|seventeen|eighteen|nineteen|twenty)\b"
            )
            safe_sentences = [
                sentence for sentence in sentences
                if not re.search(count_pattern, sentence, re.IGNORECASE)
                and not re.search(
                    r"\b(boxes|bounding|coordinates)\b",
                    sentence, re.IGNORECASE,
                )
            ]
            visible_description = " ".join(safe_sentences)
            details["description"] = description
            details["displayed_description"] = visible_description
            if visible_description:
                parts.append("**Scene description:** " + visible_description)
        except Exception as error:
            # Detection results remain usable if description generation fails.
            details["description_error"] = (
                f"{type(error).__name__}: {error}"
            )
            parts.append(
                "Detection completed, but I couldn't generate the description."
            )

    return parts, media, details


# PERSON_FACE_COVERAGE_V2
def _vf_face_coverage(person, face):
    px1, py1, px2, py2 = person
    fx1, fy1, fx2, fy2 = face
    intersection = (
        max(0, min(px2, fx2) - max(px1, fx1))
        * max(0, min(py2, fy2) - max(py1, fy1))
    )
    area = max(0, fx2 - fx1) * max(0, fy2 - fy1)
    return intersection / area if area else 0.0


def _vf_clipped_faces(persons, faces):
    clipped = []
    for face in faces:
        fx1, fy1, fx2, fy2 = face
        center_x = (fx1 + fx2) / 2
        if any(_vf_face_coverage(person, face) >= 0.9
               for person in persons):
            continue
        for px1, py1, px2, py2 in persons:
            height = py2 - py1
            if (
                height > 0
                and px1 <= center_x <= px2
                and py1 - 0.5 * height <= fy1 <= py1 + 0.3 * height
                and fy2 >= py1 - 0.2 * height
            ):
                clipped.append(face)
                break
    return clipped
