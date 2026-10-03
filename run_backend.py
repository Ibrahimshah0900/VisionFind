"""Start the VisionFind GPU backend from a repository checkout."""
from pathlib import Path
import os
import json
import getpass

ROOT = Path(__file__).resolve().parent


def build_engine():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import (
        CLIPModel, CLIPProcessor, AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )
    from src.config import MODEL_CACHE_ROOT
    from src.pipeline import VisionFind

    if not torch.cuda.is_available():
        raise RuntimeError("This backend requires a CUDA GPU.")

    settings = json.loads((ROOT / "model_config.json").read_text())
    cache = MODEL_CACHE_ROOT / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)

    def snapshot(model, revision):
        return snapshot_download(
            repo_id=model,
            revision=revision,
            cache_dir=str(cache),
            allow_patterns=[
                "*.json", "*.safetensors", "*.bin", "*.txt",
                "*.model", "*.jinja", "merges.txt",
            ],
        )

    clip_path = snapshot(
        settings["clip_model"], settings["clip_revision"]
    )
    clip_processor = CLIPProcessor.from_pretrained(
        clip_path, use_fast=False, local_files_only=True
    )
    clip = CLIPModel.from_pretrained(
        clip_path, local_files_only=True
    ).to("cpu").eval()

    vlm_path = snapshot(
        settings["vlm_model"], settings["vlm_revision"]
    )
    vlm_processor = AutoProcessor.from_pretrained(
        vlm_path,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
        use_fast=True,
        local_files_only=True,
    )
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_path,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        use_safetensors=True,
        local_files_only=True,
    ).to("cuda").eval()

    return VisionFind(ROOT, clip, clip_processor, vlm, vlm_processor)


def main():
    # Validate credentials before spending time loading models.
    token = os.environ.get("VISIONFIND_API_TOKEN") or getpass.getpass(
        "API token (hidden; use the same value in your local interface): "
    )
    if len(token) < 32:
        raise ValueError("Use a random API token of at least 32 characters.")

    import shutil
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Install FFmpeg before starting the backend.")

    os.chdir(ROOT)
    engine = build_engine()

    from src.multi_chat import build_multi_chat
    from src.api_server import create_api
    import uvicorn

    chat = build_multi_chat(engine)
    api = create_api(chat.visionfind_chat_handler, ROOT, token)
    uvicorn.run(api, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
