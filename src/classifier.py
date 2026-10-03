import numpy as np
import pandas as pd
from .embeddings import encode_query


def classify_embeddings(
    image_vectors, candidate_labels, model, processor, device,
    template="an image of {}",
):
    labels = list(candidate_labels)
    if not labels or any(not isinstance(x, str) or not x.strip() for x in labels):
        raise ValueError("Provide non-empty string labels.")
    labels = [x.strip() for x in labels]
    if len(set(labels)) != len(labels):
        raise ValueError("Candidate labels must be unique.")
    if template.count("{}") != 1:
        raise ValueError("Template must contain exactly one {}.")

    vectors = np.asarray(image_vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) == 0:
        raise ValueError("Image embeddings must be a non-empty 2D array.")
    if not np.isfinite(vectors).all():
        raise ValueError("Image embeddings must be finite.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
        raise ValueError("Image embeddings must be L2-normalized.")

    prompts = [template.format(label) for label in labels]
    text_vectors = np.stack([
        encode_query(prompt, model, processor, device)
        for prompt in prompts
    ])
    if vectors.shape[1] != text_vectors.shape[1]:
        raise ValueError("Image and text embedding dimensions differ.")

    return vectors @ text_vectors.T, prompts


def rank_labels(scores, candidate_labels, metadata, top_k=3):
    labels = [label.strip() for label in candidate_labels]
    scores = np.asarray(scores)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if scores.shape != (len(metadata), len(labels)):
        raise ValueError("Scores must align with metadata and labels.")
    if metadata["row_index"].tolist() != list(range(len(metadata))):
        raise ValueError("Metadata row order does not match the index.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")

    rows = []
    for position, (_, item) in enumerate(metadata.iterrows()):
        order = np.argsort(-scores[position], kind="stable")[:top_k]
        for rank, label_index in enumerate(order, start=1):
            rows.append({
                "row_index": int(item["row_index"]),
                "filename": item["filename"],
                "relative_path": item["relative_path"],
                "rank": rank,
                "label": labels[label_index],
                "cosine_similarity": float(scores[position, label_index]),
            })
    return pd.DataFrame(rows)
