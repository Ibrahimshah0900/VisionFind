"""Similarity scoring and image ranking for VisionFind."""

import numpy as np

def search_index(query_embedding, embeddings, metadata, top_k=3):
    """Rank images using normalized query and image embeddings."""
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("The index must be a non-empty 2D matrix.")

    if query_embedding.shape != (embeddings.shape[1],):
        raise ValueError("Query dimensions do not match the index.")

    if metadata["row_index"].tolist() != list(range(len(embeddings))):
        raise ValueError("Metadata row order does not match the index.")

    scores = embeddings @ query_embedding
    if not np.isfinite(scores).all():
        raise ValueError("Similarity scores contain invalid values.")

    top_indices = np.argsort(-scores, kind="stable")[
        :min(top_k, len(scores))
    ]

    results = metadata.iloc[top_indices].copy()
    results.insert(0, "rank", range(1, len(results) + 1))
    results["similarity"] = scores[top_indices]

    return results.reset_index(drop=True)
