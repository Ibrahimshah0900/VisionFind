import time
from pathlib import Path

import torch
from PIL import Image


@torch.inference_mode()
def answer_image(
    image_path, question, model, processor, max_new_tokens=128
):
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Provide a non-empty question.")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer.")

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question.strip()},
        ],
    }]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[prompt], images=[image],
        padding=True, return_tensors="pt",
    ).to(model.device)

    model.eval()
    on_cuda = model.device.type == "cuda"
    if on_cuda:
        torch.cuda.synchronize(model.device)
        torch.cuda.reset_peak_memory_stats(model.device)

    started = time.perf_counter()
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    if on_cuda:
        torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - started

    tokens = generated[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(
        tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    if not answer:
        raise RuntimeError("Model generated an empty answer.")

    return {
        "question": question.strip(),
        "answer": answer,
        "generation_seconds": round(elapsed, 3),
        "generated_tokens": int(tokens.shape[1]),
        "max_new_tokens": max_new_tokens,
        "peak_allocated_gpu_gib": (
            round(torch.cuda.max_memory_allocated(model.device) / 1024**3, 3)
            if on_cuda else None
        ),
    }
