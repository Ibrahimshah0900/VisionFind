"""Embedding index storage utilities for VisionFind."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

def save_index(index_dir, embeddings, metadata, settings):
    """Save an embedding matrix with its ordered metadata and settings."""
    if embeddings.ndim != 2 or len(metadata) != embeddings.shape[0]:
        raise ValueError("Each embedding row must have one metadata row.")

    if metadata["row_index"].tolist() != list(range(len(metadata))):
        raise ValueError("Metadata row_index must follow embedding order.")

    manifest = {
        "schema_version": 1,
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "settings": settings,
        "images": metadata.to_dict(orient="records"),
    }
    manifest_text = json.dumps(manifest, indent=2, allow_nan=False)

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    np.save(index_dir / "image_embeddings.npy", embeddings, allow_pickle=False)
    (index_dir / "metadata.json").write_text(manifest_text, encoding="utf-8")


def load_index(index_dir):
    """Load an index and check its dimensions and metadata row order."""
    index_dir = Path(index_dir)

    embeddings = np.load(
        index_dir / "image_embeddings.npy", allow_pickle=False
    )
    manifest = json.loads(
        (index_dir / "metadata.json").read_text(encoding="utf-8")
    )
    metadata = pd.DataFrame(manifest["images"])

    if list(embeddings.shape) != manifest["embedding_shape"]:
        raise ValueError("Saved embedding dimensions do not match metadata.")

    if str(embeddings.dtype) != manifest["embedding_dtype"]:
        raise ValueError("Saved embedding dtype does not match metadata.")

    if metadata["row_index"].tolist() != list(range(len(embeddings))):
        raise ValueError("Metadata rows do not match embedding rows.")

    return embeddings, metadata, manifest
