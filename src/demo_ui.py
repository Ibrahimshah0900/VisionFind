
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps


def build_app(engine):
    root = Path(engine.root)
    samples = {
        row["filename"]: str(root / row["relative_path"])
        for _, row in engine.metadata.iterrows()
    }
    if not samples:
        raise ValueError("The search index contains no images.")

    def new_run(feature):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        folder = root / "results/ui" / f"{stamp}_{feature}"
        folder.mkdir(parents=True, exist_ok=False)
        return folder

    def save_record(folder, record):
        record.update({
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "clip_model": engine.manifest["settings"]["model_name"],
            "clip_revision": getattr(engine.clip.config, "_commit_hash", None),
            "vlm_revision": getattr(engine.vlm.config, "_commit_hash", None),
        })
        path = folder / "result.json"
        path.write_text(
            json.dumps(record, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return str(path)

    def load_rgb(path):
        with Image.open(path) as source:
            return ImageOps.exif_transpose(source).convert("RGB")

    def search(query, top_k):
        if not query or not query.strip():
            raise gr.Error("Enter a search description.")

        started = time.perf_counter()
        results = engine.search(query.strip(), int(top_k))
        elapsed = time.perf_counter() - started

        gallery = []
        for _, row in results.iterrows():
            image = load_rgb(root / row["relative_path"])
            gallery.append((
                image,
                f"{int(row['rank'])}. {row['filename']} · "
                f"cosine {row['similarity']:.3f}",
            ))

        folder = new_run("search")
        results.to_csv(folder / "results.csv", index=False)
        export = save_record(folder, {
            "feature": "search",
            "query": query.strip(),
            "elapsed_seconds": round(elapsed, 3),
            "results": results.to_dict(orient="records"),
        })
        return (
            gallery,
            results[["rank", "filename", "similarity"]],
            f"Found {len(results)} results in {elapsed:.2f}s. Saved to Drive.",
            export,
        )

    def use_sample(name):
        return samples[name]

    def analyze(image_path, action, labels_text, question, target):
        if not image_path:
            raise gr.Error("Upload an image or load a sample first.")

        labels = []
        if action == "Classify":
            labels = [
                label.strip()
                for label in labels_text.replace("\n", ",").split(",")
                if label.strip()
            ]
            if len(labels) < 2 or len(labels) > 30:
                raise gr.Error("Enter between 2 and 30 comma-separated labels.")
            if len(set(labels)) != len(labels):
                raise gr.Error("Remove duplicate labels.")
        if action == "Ask a question" and not question.strip():
            raise gr.Error("Enter a question.")
        if action == "Locate an object" and not target.strip():
            raise gr.Error("Describe the object to locate.")

        folder = new_run("image")
        image = load_rgb(image_path)
        saved_image = folder / "input.png"
        image.save(saved_image)

        started = time.perf_counter()
        record = {
            "feature": action,
            "input_image": "input.png",
        }
        table = None
        output_image = image

        if action == "Classify":
            ranked = engine.classify(
                saved_image, labels, top_k=min(5, len(labels))
            )
            table = ranked[["rank", "label", "cosine_similarity"]]
            table.to_csv(folder / "classification.csv", index=False)
            record["labels"] = labels
            record["predictions"] = table.to_dict(orient="records")
            best = table.iloc[0]
            answer = (
                f"Top match: {best['label']}\n"
                f"Cosine similarity: {best['cosine_similarity']:.4f}\n\n"
                "Scores rank your candidate labels; they are not probabilities."
            )

        elif action == "Caption":
            result = engine.caption(saved_image)
            record["result"] = result
            answer = result["answer"]

        elif action == "Ask a question":
            result = engine.ask(saved_image, question.strip())
            record["result"] = result
            answer = result["answer"]

        elif action == "Locate an object":
            result, output_image = engine.locate(
                saved_image, target.strip()
            )
            record["result"] = result
            output_image.save(folder / "grounding.png")

            if result["status"] == "valid_coordinates":
                answer = (
                    f"Target: {target.strip()}\n"
                    f"Box in original image pixels: "
                    f"{[round(v, 1) for v in result['bbox_original']]}\n\n"
                    "Approximate model prediction; visually check the box."
                )
            elif result["status"] == "target_reported_absent":
                answer = "The model reported that the target was not present."
            else:
                answer = (
                    "The model did not return a usable box.\n"
                    f"Reason: {result.get('error', 'Invalid output')}\n"
                    f"Raw response: {result['raw_output']}"
                )
        else:
            raise gr.Error("Choose a supported action.")

        elapsed = time.perf_counter() - started
        record["elapsed_seconds"] = round(elapsed, 3)
        export = save_record(folder, record)

        return (
            answer,
            output_image,
            gr.update(value=table, visible=table is not None),
            f"Completed in {elapsed:.2f}s. Input and results saved to Drive.",
            export,
        )

    css = """
    .gradio-container {max-width: 1180px !important; margin: auto;}
    #hero {padding: 24px; border-radius: 18px;
           background: linear-gradient(120deg, #101c35, #214c62);
           margin-bottom: 18px;}
    #hero h1, #hero p {color: white !important;}
    #hero h1 {font-size: 36px; margin-bottom: 8px;}
    """

    with gr.Blocks(
        title="VisionFind",
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="slate"),
        css=css,
        delete_cache=(3600, 86400),
    ) as demo:
        gr.HTML("""
        <div id="hero">
          <h1>VisionFind</h1>
          <p>Find images with words. Explore what’s inside them.</p>
        </div>
        """)

        with gr.Tab("Search"):
            gr.Markdown(
                f"Search the **{len(samples)} images** in your saved collection."
            )
            with gr.Row():
                query = gr.Textbox(
                    label="Describe an image",
                    value="a cup of coffee",
                    placeholder="Try: an astronaut wearing a spacesuit",
                    scale=4,
                )
                top_k = gr.Slider(
                    minimum=1, maximum=len(samples),
                    value=min(3, len(samples)), step=1,
                    label="Results", scale=1,
                )
            search_button = gr.Button("Find images", variant="primary")
            gallery = gr.Gallery(
                label="Matching images", columns=3, height=360,
                object_fit="contain",
            )
            search_table = gr.Dataframe(
                headers=["rank", "filename", "similarity"],
                interactive=False, label="Search scores",
            )
            search_status = gr.Textbox(label="Status", interactive=False)
            search_export = gr.File(label="Download search record")
            gr.Examples(
                examples=[
                    ["a cup of coffee"],
                    ["an astronaut wearing a spacesuit"],
                    ["a close-up photo of a cat"],
                ],
                inputs=[query],
            )

        with gr.Tab("Image Studio"):
            gr.Markdown(
                "Upload your own image or load a sample. "
                "Studio uploads are analyzed individually; "
                "they are not added to the search collection."
            )
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        type="filepath", sources=["upload"],
                        label="Image", height=350,
                    )
                    with gr.Row():
                        sample = gr.Dropdown(
                            choices=list(samples),
                            value=next(iter(samples)),
                            label="Sample image",
                        )
                        sample_button = gr.Button("Load sample")
                    action = gr.Radio(
                        ["Caption", "Ask a question", "Classify", "Locate an object"],
                        value="Caption", label="Action",
                    )
                    labels = gr.Textbox(
                        label="Classification labels",
                        value="cat, dog, cup of coffee, person, horse, camera, coins",
                        lines=2,
                    )
                    question = gr.Textbox(
                        label="Question",
                        value="What is the main subject in this image?",
                    )
                    target = gr.Textbox(
                        label="Object to locate",
                        value="the main object",
                    )
                    run_button = gr.Button("Analyze image", variant="primary")

                with gr.Column():
                    answer = gr.Textbox(
                        label="Result", lines=6, interactive=False,
                    )
                    output_image = gr.Image(
                        label="Image / predicted box", height=350,
                    )
                    classification = gr.Dataframe(
                        headers=["rank", "label", "cosine_similarity"],
                        visible=False, interactive=False,
                        label="Label ranking",
                    )
                    image_status = gr.Textbox(label="Status", interactive=False)
                    image_export = gr.File(label="Download result record")

            gr.Markdown(
                "Successful analyses save the input image and results to your "
                "project in Drive. Captions and answers can contain mistakes; "
                "bounding boxes are approximate."
            )

        with gr.Tab("About"):
            gr.Markdown("""
### VisionFind
- **CLIP:** semantic image search and custom-label classification.
- **Qwen2.5-VL:** captions, image questions, and approximate localization.
- Six-image retrieval check: expected image ranked first for 6/6 queries.
- Classification: 6/6 expected labels for each of three prompt styles.
- Caption review found incorrect and unsupported details.
- Three grounding examples were visually acceptable; IoU was not measured.

These are small development checks, not a general accuracy benchmark.
""")

        # All inference actions share one queue to avoid concurrent GPU calls.
        event_options = {
            "concurrency_limit": 1,
            "concurrency_id": "visionfind_inference",
            "api_name": False,
        }
        search_button.click(
            search, [query, top_k],
            [gallery, search_table, search_status, search_export],
            **event_options,
        )
        query.submit(
            search, [query, top_k],
            [gallery, search_table, search_status, search_export],
            **event_options,
        )
        sample_button.click(
            use_sample, sample, image_input,
            queue=False, api_name=False,
        )
        run_button.click(
            analyze,
            [image_input, action, labels, question, target],
            [answer, output_image, classification, image_status, image_export],
            **event_options,
        )

    return demo
