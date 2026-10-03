"""Image and text embedding utilities for VisionFind."""

import numpy as np
import torch
from PIL import Image

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Return a float32 array with each embedding normalized to length 1."""
    vectors = np.asarray(embeddings, dtype=np.float32)

    if vectors.ndim != 2 or 0 in vectors.shape:
        raise ValueError(
            "Expected a non-empty 2D array: (number_of_items, dimensions)."
        )

    lengths = np.linalg.norm(
        vectors, ord=2, axis=1, keepdims=True
    )

    if not np.isfinite(lengths).all() or np.any(lengths == 0):
        raise ValueError("Embeddings must have finite, nonzero lengths.")

    return vectors / lengths


@torch.inference_mode()
def encode_images(image_paths, model, processor, device, batch_size=4):
    """Return normalized embeddings in the same order as image_paths."""
    if not image_paths:
        raise ValueError("Provide at least one image path.")

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    model.eval()
    embedding_batches = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        batch_images = []

        for path in batch_paths:
            with Image.open(path) as image:
                batch_images.append(image.convert("RGB"))

        inputs = processor(
            images=batch_images,
            return_tensors="pt",
        ).to(device)

        features = model.get_image_features(**inputs)
        embeddings = normalize_embeddings(
            features.detach().cpu().numpy()
        )
        embedding_batches.append(embeddings)

    return np.concatenate(embedding_batches, axis=0)


@torch.inference_mode()
def encode_query(query, model, processor, device):
    """Encode one non-empty query as a normalized 1D NumPy vector."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Enter a non-empty text query.")

    inputs = processor(
        text=[query.strip()],
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    token_limit = model.config.text_config.max_position_embeddings
    if inputs["input_ids"].shape[1] > token_limit:
        raise ValueError(f"Query exceeds CLIP's {token_limit}-token limit.")

    model.eval()
    features = model.get_text_features(**inputs.to(device))

    return normalize_embeddings(features.detach().cpu().numpy())[0]
