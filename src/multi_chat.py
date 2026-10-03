
import copy
import hashlib
import json
import re
import uuid
from pathlib import Path

import gradio as gr
import torch
from PIL import Image, ImageOps

from . import query_router as router
from .general_detector import resolve_target as resolve_general_target, chat_reply as general_detector_reply
from .chat_ui import detector_kind, detector_reply, locate_many, needs_text_answer


def save_json(path, value):
    """Write via a temporary file so interrupted writes preserve prior progress."""
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(path)


def read_image(path):
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
IMAGE_WORD = r"(?:images?|photos?|pictures?|pics?)"
NUMBER_REFERENCE = (
    rf"\b{IMAGE_WORD}\s*#?\s*\d+"
    rf"(?:\s*(?:,|and|&|to|-)\s*(?:{IMAGE_WORD}\s*)?#?\s*\d+)*"
)


def select_images(message, available, previous, newly_added):
    """Resolve explicit references before using conversation context."""
    lower = message.lower()
    ids = []
    for match in re.finditer(NUMBER_REFERENCE, lower):
        reference = match.group()
        numbers = [int(n) for n in re.findall(r"\d+", reference)]
        if re.search(r"\d+\s*(?:to|-)\s*\d+", reference):
            if len(numbers) != 2 or numbers[1] < numbers[0]:
                raise ValueError("Use an ascending range, such as images 1 to 3.")
            if numbers[1] - numbers[0] > 10:
                raise ValueError("Choose image numbers within this conversation.")
            numbers = list(range(numbers[0], numbers[1] + 1))
        ids.extend(numbers)

    for word, number in ORDINALS.items():
        if re.search(rf"\b{word}\s+{IMAGE_WORD}\b", lower):
            ids.append(number)
    if re.search(rf"\blast\s+{IMAGE_WORD}\b", lower) and available:
        ids.append(available[-1])

    compare = bool(re.search(
        r"\b(compare|comparison|differences?|similarities)\b"
        r"|\bwhich\s+(?:image|photo|picture)"
        r"|\b(?:similar|different|same)\b.*\b(?:images|photos|pictures|pics)\b",
        lower,
    ))
    all_requested = bool(re.search(
        rf"\b(?:all|each|every|these|both)\s+(?:the\s+)?{IMAGE_WORD}\b"
        r"|\ball of them\b|\ball\s+(?:uploaded|attached)\s+(?:images|photos|pictures|pics)\b",
        lower,
    ))

    # A quantity is not an image ID: “three images” differs from “image 3”.
    quantity = re.search(
        rf"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        rf"\s+(?:of\s+(?:the\s+)?)?{IMAGE_WORD}\b", lower,
    )
    if quantity and not ids:
        words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
        token = quantity.group(1)
        count = int(token) if token.isdigit() else words.index(token) + 1
        candidates = list(newly_added or available)
        if count != len(candidates):
            raise ValueError("Please specify the image numbers; that quantity does not identify a unique selection.")
        ids = candidates
    if re.search(r"\b(?:the|attached)\s+(?:images|photos|pictures|pics)\b", lower):
        all_requested = True

    if not ids:
        if all_requested or compare:
            ids = list(available)
        elif newly_added:
            ids = list(newly_added)
        elif previous:
            ids = [n for n in previous if n in available]
        elif len(available) == 1:
            ids = list(available)
        elif available:
            raise ValueError(
                "Which image should I use? Say “image 2” or “all images”."
            )

    if re.search(r"\bboth\s+(?:the\s+)?(?:images|photos|pictures|pics)\b", lower) and len(ids) != 2:
        raise ValueError("Which two images do you mean? Use their image numbers.")
    ids = list(dict.fromkeys(ids))
    missing = [n for n in ids if n not in available]
    if missing:
        raise ValueError(f"Image numbers not available: {missing}.")
    if compare and len(ids) < 2:
        raise ValueError("Choose at least two images to compare.")

    # Remove image selectors before passing to the single-image action router.
    cleaned = re.sub(NUMBER_REFERENCE, "the image", message, flags=re.I)
    for word in list(ORDINALS) + ["last"]:
        cleaned = re.sub(
            rf"\b{word}\s+{IMAGE_WORD}\b", "the image", cleaned, flags=re.I
        )
    cleaned = re.sub(
        rf"\b(?:all|each|every|these|both)\s+(?:the\s+)?{IMAGE_WORD}\b",
        "the image", cleaned, flags=re.I,
    )
    cleaned = re.sub(
        rf"\b(?:the\s+)?(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        rf"\s+(?:of\s+(?:the\s+)?)?{IMAGE_WORD}(?:\s+attached)?\b",
        "the image", cleaned, flags=re.I,
    )
    cleaned = re.sub(r"\b(?:the|attached)\s+(?:images|photos|pictures|pics)\b",
                     "the image", cleaned, flags=re.I)
    cleaned = re.sub(r"\ball\s+(?:uploaded|attached)\s+(?:images|photos|pictures|pics)\b",
                     "the image", cleaned, flags=re.I)
    return ids, compare, cleaned


@torch.inference_mode()
def compare_images(engine, items, question):
    # 448x448 padded images provide 256 visual tokens each with this processor.
    images, content = [], []
    for item in items:
        image = read_image(item["path"])
        image.thumbnail((448, 448), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (448, 448), "white")
        canvas.paste(image, ((448-image.width)//2, (448-image.height)//2))
        images.append(canvas)
        content.extend([
            {"type": "text", "text": f"Image {item['id']}:"},
            {"type": "image"},
        ])

    content.append({
        "type": "text",
        "text": (
            "Answer using these actual images. Refer to their supplied image "
            "numbers, not their position in this prompt. State uncertainty. "
            "Do not invent details or claim that boxes were drawn. "
            "Request: " + question
        ),
    })
    processor = engine.vlm_processor
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[prompt], images=images, padding=True, return_tensors="pt"
    ).to(engine.vlm.device)
    output = engine.vlm.generate(
        **inputs, max_new_tokens=240, do_sample=False
    )
    answer = processor.batch_decode(
        output[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return answer


def analyze_one(engine, item, message, plan, folder, history):
    image_path = item["path"]
    media = []
    details = {"image_id": item["id"], "plan": plan}

    if plan.get("locate") and re.fullmatch(
        r"(?:the |all |any |visible |main )*(?:objects?|things?|items?)"
        r"(?: found)?(?: in (?:the )?image)?", plan.get("target", "").strip(), re.I,
    ):
        details["needs_clarification"] = True
        return ("Which objects should I box—people, faces, dogs, or another target? "
                "You can name more than one target.", [], details)

    if detector_kind(plan, message):
        parts, media, result = detector_reply(
            engine, image_path, message, plan, folder
        )
        details["detector_result"] = result
        return "\n\n".join(parts), media, details

    if (plan.get("mode") == "answer" and plan.get("target")
            and (plan.get("locate") or re.search(r"\b(how many|count|number of)\b", message, re.I))
            and resolve_general_target(plan["target"])):
        text, media, result = general_detector_reply(engine, image_path, message, plan, folder)
        details["detector_result"] = result
        return text, media, details

    parts = []
    if plan["mode"] == "classify":
        table = engine.classify(image_path, plan["labels"])
        details["classification"] = table.to_dict(orient="records")
        parts.append(
            "Label ranking:\n" +
            "\n".join(
                f"- {row['label']}: {row['cosine_similarity']:.3f}"
                for _, row in table.iterrows()
            ) + "\nCosine scores are not probabilities."
        )
    elif not plan["locate"] or needs_text_answer(message):
        # Router history resolves references; old generated descriptions are
        # not visual evidence and must not contaminate a fresh image answer.
        visual_request = message
        if re.fullmatch(r"\s*(?:describe|descirbe|explain)\s+(?:the\s+)?image[.!?\s]*", message, re.I):
            visual_request = "Describe the visible contents of this image."
        details["visual_request"] = visual_request
        result = engine.ask(
            image_path,
            "You have exactly one attached image. Describe only what is visible in it. "
            "Do not discuss other images or invent additional scenes. "
            "Do not output coordinates or claim to draw boxes; a separate "
            "tool handles drawing. State uncertainty when needed.\n"
            f"Request: {visual_request}",
            max_new_tokens=200,
        )
        details["answer"] = result
        parts.append(result["answer"])

    if plan["locate"]:
        result, annotated = locate_many(
            engine, image_path, plan["target"], folder
        )
        details["grounding"] = result
        if annotated:
            count = len(result["valid_boxes"])
            parts.append(f"Drew {count} approximate predicted box(es).")
            media.append((annotated, "Predicted locations"))
        else:
            parts.append(
                "No usable boxes were returned. This does not prove "
                "the target is absent."
            )
    return "\n\n".join(parts), media, details


def build_multi_chat(engine):
    root = Path(engine.root)

    def respond_single(message, uploads, chat, state, progress=gr.Progress()):
        message = (message or "").strip()
        uploads = uploads or []
        if not message and not uploads:
            raise gr.Error("Type a message or attach images.")
        if len(message) > 2000:
            raise gr.Error("Keep messages below 2,000 characters.")
        if len(uploads) > 10:
            raise gr.Error("Upload at most 10 images at a time.")

        state = copy.deepcopy(state or {})
        chat = list(chat or [])
        state.setdefault("session_id", uuid.uuid4().hex)
        state.setdefault("images", [])
        state.setdefault("history", [])
        state.setdefault("per_image_history", {})

        session = root / "results/multi_chat" / state["session_id"]
        session.mkdir(parents=True, exist_ok=True)

        # Validate the entire upload batch before adding it.
        prepared = []
        uploaded_hashes = []
        known = {item["hash"] for item in state["images"]}
        for uploaded in uploads:
            image = read_image(uploaded)
            digest = hashlib.sha256(
                str(image.size).encode() + image.tobytes()
            ).hexdigest()
            uploaded_hashes.append(digest)
            if digest not in known:
                prepared.append((str(uploaded), image, digest))
                known.add(digest)
        if len(state["images"]) + len(prepared) > 10:
            raise gr.Error(
                "This conversation would exceed 10 unique images. "
                "Start a new conversation from Help."
            )

        new_ids = []
        for original_path, image, digest in prepared:
            number = len(state["images"]) + 1
            saved = session / f"image_{number:02d}.png"
            image.save(saved)
            item = {
                "id": number, "path": str(saved), "hash": digest,
                "filename": Path(original_path).name,
            }
            state["images"].append(item)
            new_ids.append(number)
        # Originals remain intact for inference. The chat only renders thumbnails.
        attached = [next(item for item in state["images"] if item["hash"] == digest)
                    for digest in dict.fromkeys(uploaded_hashes)]
        attached_ids = [item["id"] for item in attached]
        if attached:
            chat.append({"role": "user", "content": gr.Gallery(
                value=[(item["path"], f"Image {item['id']}") for item in attached],
                columns=min(5, len(attached)), rows=1 if len(attached) <= 5 else 2,
                height=126 if len(attached) <= 5 else 244,
                object_fit="contain", show_label=False, container=False,
                allow_preview=True, show_download_button=False,
                elem_classes=["upload-thumbnails"],
            )})

        save_json(session / "session.json", state)
        message = message or "Describe these images."
        chat.append({"role": "user", "content": message})
        turn_dir = session / uuid.uuid4().hex
        turn_dir.mkdir()
        log = {"message": message, "results": [], "status": "running"}
        text_replies = []
        # Render the submitted message before waiting for model generation.
        yield "", None, chat, state, []

        def add_reply(text):
            text_replies.append(text)
            chat.append({"role": "assistant", "content": text})

        try:
            available = [item["id"] for item in state["images"]]

            # Uploaded photos are the default search scope when present.
            search_request = bool(re.search(
                r"\b(search|find|retrieve)\b.*\b(photos?|images?|pictures?|collection)\b",
                message, re.I,
            ))
            explicit_collection = bool(re.search(
                r"\b(?:(?:existing|indexed|sample|saved)\s+collection|sample\s+(?:photos|images))\b",
                message, re.I,
            ))
            uploaded_search = search_request and bool(available) and not explicit_collection
            collection_search = search_request and not uploaded_search

            if uploaded_search:
                from .upload_rag import UploadedImageRAG
                plan, audit = router.route_request(engine, message, state["history"], True)
                log["routing"] = audit
                if plan["mode"] != "search" or not plan.get("query", "").strip():
                    add_reply(plan.get("clarification") or "What should I search for in your uploaded photos?")
                else:
                    rag = UploadedImageRAG.from_state(engine, state)
                    matches = rag.search(plan["query"], top_k=3)
                    log["results"] = [{"operation": "uploaded_image_retrieval", "plan": plan,
                        "matches": matches, "cache_status": rag.cache_status}]
                    state["selected_ids"] = [matches[0]["image_id"]]
                    add_reply("Ranked candidates from your uploaded images:\n" + "\n".join(
                        f"- Image {m['image_id']} — cosine {m['similarity']:.3f}" for m in matches
                    ) + "\nThese are similarity matches, not confirmed object detections. Ask about an image by its number.")
                    chat.append({"role": "assistant", "content": gr.Gallery(
                        value=[(m["path"], f"Image {m['image_id']}") for m in matches],
                        columns=min(3, len(matches)), height=126, show_label=False,
                        allow_preview=True, elem_classes=["upload-thumbnails"],
                    )})
            elif collection_search:
                plan, audit = router.route_request(
                    engine, message, state["history"], bool(available)
                )
                log["routing"] = audit
                if plan["mode"] == "search":
                    matches = engine.search(plan["query"], top_k=3)
                    log["results"] = matches.to_dict(orient="records")
                    add_reply("Matches from the existing indexed collection:")
                    for _, row in matches.iterrows():
                        chat.extend([
                            {"role": "assistant", "content":
                             f"{row['filename']} — cosine {row['similarity']:.3f}"},
                            {"role": "assistant", "content":
                             gr.Image(value=str(root / row["relative_path"]))},
                        ])
                else:
                    add_reply(
                        plan.get("clarification") or
                        "Please describe which photos you want to search for."
                    )
            else:
                ids, comparison, cleaned = select_images(
                    router.normalize_query(message), available, state.get("selected_ids", []), attached_ids
                )
                if not ids:
                    raise ValueError("Attach the image(s) you want me to analyze.")

                selected = [
                    next(item for item in state["images"] if item["id"] == n)
                    for n in ids
                ]
                state["selected_ids"] = ids
                log["selected_ids"] = ids
                log["single_image_request"] = cleaned

                if comparison:
                    # Do not silently drop a second action from a mixed request.
                    if re.search(
                        r"\b(box|boxes|locate|highlight|outline)\b", message, re.I
                    ):
                        raise ValueError(
                            "For this version, compare the images first, then "
                            "ask for boxes on the selected images."
                        )
                    progress(0.2, desc="Comparing selected images")
                    answer = compare_images(engine, selected, message)
                    add_reply(answer)
                    log["results"].append({
                        "image_ids": ids,
                        "operation": "joint_visual_comparison",
                        "answer": answer,
                    })
                else:
                    # Each image has its own history to prevent referent leakage.
                    for position, item in enumerate(selected):
                        number = item["id"]
                        progress(
                            position / len(selected),
                            desc=f"Processing Image {number}",
                        )
                        folder = turn_dir / f"image_{number:02d}"
                        folder.mkdir()
                        history = state["per_image_history"].get(str(number), [])
                        try:
                            plan, audit = router.route_request(
                                engine, cleaned, history, True
                            )
                            if plan["mode"] == "clarify":
                                text = plan["clarification"]
                                media = []
                                detail = {"plan": plan, "image_id": number}
                            elif plan["mode"] == "search":
                                text = (
                                    "Searching uploaded photos is the next RAG "
                                    "phase. Ask about a numbered image or compare "
                                    "the selected images for now."
                                )
                                media = []
                                detail = {"plan": plan, "image_id": number}
                            else:
                                text, media, detail = analyze_one(
                                    engine, item, cleaned, plan, folder, history
                                )
                            detail["routing"] = audit
                            add_reply(f"**Image {number}**\n\n{text}")
                            for media_path, caption in media:
                                chat.extend([
                                    {"role": "assistant", "content":
                                     f"Image {number} — {caption}"},
                                    {"role": "assistant", "content":
                                     gr.Image(value=media_path, height=420, show_label=False)},
                                ])
                            history.extend([
                                {"role": "user", "content": cleaned},
                                {"role": "assistant", "content": text[:1800]},
                            ])
                            state["per_image_history"][str(number)] = history[-6:]
                        except Exception as error:
                            detail = {
                                "image_id": number,
                                "error": f"{type(error).__name__}: {error}",
                            }
                            add_reply(
                                f"**Image {number}:** couldn't complete this "
                                f"analysis. {type(error).__name__}: {str(error)[:160]}"
                            )
                            if isinstance(error, torch.cuda.OutOfMemoryError):
                                torch.cuda.empty_cache()

                        log["results"].append(detail)
                        save_json(folder / "result.json", detail)
                        save_json(turn_dir / "turn.json", log)
                        save_json(session / "session.json", state)
                        yield "", None, chat, state, []

            results = log["results"]
            attention = [r for r in results if isinstance(r, dict) and (
                r.get("error") or r.get("needs_clarification") or
                r.get("plan", {}).get("mode") == "clarify")]
            log["status"] = ("partial" if len(attention) < len(results) else "needs_attention") if attention else "completed"
        except Exception as error:
            log["status"] = "needs_attention"
            log["error"] = f"{type(error).__name__}: {error}"
            if isinstance(error, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                add_reply(
                    "The comparison exceeded available GPU memory. "
                    "Try comparing two or three images."
                )
            else:
                add_reply(str(error))

        state["history"].extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": "\n".join(text_replies)[:2400]},
        ])
        state["history"] = state["history"][-8:]
        save_json(turn_dir / "turn.json", log)
        save_json(session / "session.json", state)

        gallery = [
            (item["path"], f"Image {item['id']}")
            for item in state["images"]
        ]
        yield "", None, chat, state, gallery


    def respond(payload, chat, state, progress=gr.Progress()):
        """Text and files arrive atomically through the native composer."""
        payload = payload or {}
        message = (payload.get("text") or "").strip()
        uploads = payload.get("files") or []
        # CAPABILITY_HELP_V1
        normalized = re.sub(r"[^a-z0-9 ]", "", message.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = re.sub(r"^(?:hi|hello|hey|bro)\s+", "", normalized)
        help_request = re.fullmatch(
            r"(?:what (?:can you do|you can do)(?: for me)?|"
            r"what can you help (?:me )?with|"
            r"(?:do )?you only analy[sz]e (?:the )?images?|"
            r"(?:do )?you only analy[sz]e photos|"
            r"(?:hi|hello|hey|thanks|thank you))", normalized)
        if not uploads and help_request:
            reply = (
                "I can describe and compare your photos, answer visual questions, "
                "search your uploaded images, and draw boxes around supported objects. "
                "You can attach up to 10 photos or one short video. "
                "For videos, I can return processed object boxes and detection counts "
                "and timestamps. You can also dictate into the editable chat box. "
                "Descriptions and detections can make mistakes. What would you like to try?"
            )
            updated_chat = list(chat or []) + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply}
            ]
            yield gr.update(value={"text": "", "files": []}, interactive=True), updated_chat, state or {}
            return
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
        has_video_upload = any(
            Path((item.get('path') or item.get('name') or '') if isinstance(item, dict)
                 else str(item)).suffix.lower() in video_extensions
            for item in uploads
        )
        saved_state = state or {}
        video_followup = bool(not uploads and saved_state.get('video') and
            ('video' in message.lower() or not saved_state.get('images')))
        if has_video_upload or video_followup:
            from src import video_chat
            yield from video_chat.respond(engine, payload, chat, state)
            return
        requests = split_requests(message)
        if not requests:
            requests = ["Describe these images."] if uploads else []
        if not requests:
            raise gr.Error("Type a message or attach photos.")
        if len(message) > 2000:
            raise gr.Error("Keep messages below 2,000 characters.")
        if len(requests) > 8:
            raise gr.Error("Please send at most 8 requests in one message.")

        current_chat = list(chat or [])
        current_state = state or {}
        remaining_uploads = uploads
        # Prevent a second submit or an attachment edit while this batch runs.
        yield gr.update(interactive=False), current_chat, current_state
        for index, request in enumerate(requests):
            progress(index / len(requests), desc=f"Request {index + 1} of {len(requests)}")
            try:
                for result in respond_single(request, remaining_uploads, current_chat, current_state, progress):
                    _, _, current_chat, current_state, _ = result
                    remaining_uploads = []
                    yield gr.update(value={"text": "", "files": []}, interactive=False), current_chat, current_state
            except Exception as error:
                current_chat.append({"role": "assistant", "content": f"I couldn't complete this request: {error}"})
                if remaining_uploads:
                    yield gr.update(value=payload, interactive=True), current_chat, current_state
                    return
        yield gr.update(value={"text": "", "files": []}, interactive=True), current_chat, current_state

    def transcribe_voice(audio_path, payload):
        """Populate editable composer; never execute a chat request."""
        from . import voice
        import shutil
        original_payload = copy.deepcopy(payload or {"text": "", "files": []})
        yield gr.update(interactive=False), gr.update()
        folder = root / "results/voice_checks" / uuid.uuid4().hex
        folder.mkdir(parents=True, exist_ok=False)
        try:
            if not audio_path:
                raise ValueError("Record a spoken query first.")
            source = Path(audio_path)
            if source.stat().st_size > voice.MAX_BYTES:
                raise ValueError("Audio exceeds the 20 MiB limit.")
            saved_audio = folder / ("input_audio" + source.suffix.lower())
            shutil.copyfile(source, saved_audio)
            transcript = voice.transcribe_audio(saved_audio)
            edited_payload = voice.merge_transcript(original_payload, transcript)
            if len(edited_payload.get("text", "")) > 2000:
                raise ValueError("Combined text exceeds 2,000 characters; shorten it before adding voice.")
            save_json(folder / "transcript.json", transcript)
            gr.Info("Transcript added. Review or correct it, then press Send.")
            yield gr.update(value=edited_payload, interactive=True), gr.update(value=None, visible=False)
        except Exception as error:
            save_json(folder / "error.json", {"error": f"{type(error).__name__}: {error}"})
            yield gr.update(value=original_payload, interactive=True), gr.update()
            gr.Warning(f"Voice transcription failed: {str(error)[:160]}")

    css = """
    .gradio-container { max-width: 920px !important; margin: auto !important; }
    #vision-chat { border: none !important; background: transparent !important; box-shadow: none !important; }
    #vision-chat .bubble-wrap { background: transparent !important; }
    #vision-chat .message { font-size: 15px; line-height: 1.65; }
    #vision-composer { border-radius: 26px !important; box-shadow: 0 3px 18px #0000000a; }
    #vision-composer textarea { border: none !important; box-shadow: none !important; font-size: 16px; }
    #vision-chat .upload-thumbnails { max-width: 620px; }
    #vision-chat .upload-thumbnails .gallery-item { border-radius: 12px; }
    footer { display: none !important; }
    @media (max-width: 640px) {
        .gradio-container { padding: 8px !important; }
        #vision-chat .message { font-size: 14px; }
    }
    """

    with gr.Blocks(title="VisionFind", theme=gr.themes.Soft(primary_hue="slate"),
                   css=css, fill_height=True) as app:
        state = gr.State({})
        gr.Markdown("### VisionFind")
        with gr.Tab("Chat"):
            chat = gr.Chatbot(type="messages", height="65vh", show_label=False,
                              placeholder="What would you like to know about your photos?",
                              elem_id="vision-chat", layout="bubble", show_copy_button=True)
            with gr.Row():
                composer = gr.MultimodalTextbox(
                    file_count="multiple", file_types=["image", "video"], sources=["upload"],
                    placeholder="Ask about your photos…", show_label=False,
                    lines=1, max_lines=6, submit_btn=True, interactive=True,
                    elem_id="vision-composer", scale=1,
                )
                microphone = gr.Button("🎙", scale=0, min_width=44, size="sm",
                                       elem_id="vision-microphone")
            recorder = gr.Audio(sources=["microphone"], type="filepath", format="wav",
                                show_label=False, visible=False, elem_id="vision-recorder")
        with gr.Tab("Help"):
            gr.Markdown("""
### Using the chat
Use **🎙** to open the recorder. Stop recording to add an editable transcript.
Review object names before pressing Send; transcription can mishear words.
Voice recordings and transcripts are saved with project results in Drive.
Attach up to **10 unique images** per conversation with **＋**. Review or remove
attachments inside the composer before sending. **Enter sends; Shift+Enter adds a line.**
Uploaded images appear as small thumbnails; click one to inspect it. Box results appear larger.

Use image numbers to choose photos. You can send several requests together:

Describe all images.  
Compare images 1 and 2.  
Draw boxes around all people in image 3.

Separate actions with sentences or new lines. Follow-ups use the last selected images.
### Current limits
- Results and originals are saved in Drive; reopening a browser starts a new chat.
- Search uses the existing indexed collection, not a new index of your uploads.
- Detected counts can miss or duplicate subjects. Descriptions and boxes can be wrong.
- General-object boxes are approximate. Name the objects you want located.
- Voice input and video tracking are still planned.
""")
            reset = gr.Button("New conversation", size="sm")
            gr.Markdown("This clears the conversation view, not saved Drive files.")

        options = dict(concurrency_limit=1, concurrency_id="visionfind_multi", api_name=False)
        microphone.click(lambda: gr.update(visible=True), outputs=recorder, queue=False)
        recorder.stop_recording(transcribe_voice, [recorder, composer], [composer, recorder],
                                trigger_mode="once", **options)
        composer.submit(respond, [composer, chat, state], [composer, chat, state],
                        trigger_mode="once", **options)
        reset.click(lambda: ({"text": "", "files": []}, [], {}),
                    outputs=[composer, chat, state], **options)
    # Notebook tests can call the same handler without launching a server.
    app.visionfind_voice_handler = transcribe_voice
    app.visionfind_chat_handler = respond
    return app


def split_requests(message):
    """Split explicit action clauses without breaking 'images 1 and 2'."""
    text = message.strip()
    if not text:
        return []

    verbs = (
        r"describe|descirbe|explain|compare|draw|locate|highlight|"
        r"classify|count|find|search|tell"
    )
    # Separate lines, semicolons, or a new imperative after sentence punctuation.
    text = re.sub(
        rf"(?<=[.!?])\s+(?=(?:please\s+)?(?:{verbs})\b)",
        "\n", text, flags=re.I,
    )
    # Handle "describe all images and then compare images 1 and 2".
    text = re.sub(
        rf"\s*(?:,\s*)?\b(?:and then|then|and also|also|and)\s+"
        rf"(?=(?:please\s+)?(?:{verbs})\b)",
        "\n", text, flags=re.I,
    )
    parts = re.split(r"[\n;]+", text)
    clauses = [
        re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", part).strip()
        for part in parts if part.strip()
    ]
    # Coalesce only explicit adjacent count/box clauses with identical targets
    # and image qualifiers. Different targets/selections stay independent.
    def target_key(value):
        value = re.sub(r"\s+", " ", value.lower()).strip(" .?!")
        return re.sub(r"^(?:(?:the|all|visible|each|every) )+", "", value)

    merged = []
    i = 0
    while i < len(clauses):
        if i + 1 < len(clauses):
            count = re.fullmatch(r"(?:please )?count\s+(.+)", clauses[i], re.I)
            boxes = re.fullmatch(
                r"(?:please )?(?:draw|add)\s+(?:bounding\s+)?boxes?\s+(?:around|for)\s+(.+)",
                clauses[i+1], re.I,
            )
            if count and boxes and (
                    target_key(count[1]) == target_key(boxes[1])
                    or target_key(boxes[1]) == "them"
            ):
                box_clause = clauses[i+1]
                if target_key(boxes[1]) == "them":
                    # Resolve only this adjacent explicit count target, before
                    # asking the router. No guessed target or history needed.
                    box_clause = box_clause[:boxes.start(1)] + count[1].strip(" .?!")
                merged.append(clauses[i].rstrip(" .?!") + " and " + box_clause)
                i += 2
                continue
        merged.append(clauses[i])
        i += 1
    return merged
