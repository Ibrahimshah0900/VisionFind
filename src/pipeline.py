from pathlib import Path

import numpy as np
import pandas as pd

from .index import load_index
from .embeddings import encode_images, encode_query
from .search import search_index
from .classifier import classify_embeddings, rank_labels
from .vqa import answer_image
from .grounding import locate_object


class VisionFind:
    def __init__(
        self, project_root, clip_model, clip_processor,
        vlm_model, vlm_processor,
    ):
        self.root = Path(project_root)
        self.clip = clip_model
        self.clip_processor = clip_processor
        self.vlm = vlm_model
        self.vlm_processor = vlm_processor

        self.embeddings, self.metadata, self.manifest = load_index(
            self.root / "data/index"
        )
        settings = self.manifest["settings"]
        if settings.get("normalization") != "l2":
            raise ValueError("Expected an L2-normalized index.")
        if not np.isfinite(self.embeddings).all() or not np.allclose(
            np.linalg.norm(self.embeddings, axis=1), 1.0, atol=1e-5
        ):
            raise ValueError("Invalid or unnormalized index embeddings.")

        revision = settings.get("model_revision")
        if revision and revision != getattr(self.clip.config, "_commit_hash", None):
            raise ValueError("CLIP revision differs from the saved index.")

    def search(self, query, top_k=3):
        vector = encode_query(
            query, self.clip, self.clip_processor, self.clip.device
        )
        return search_index(vector, self.embeddings, self.metadata, top_k)

    def classify(self, image_path, labels, top_k=3, template="an image of {}"):
        path = Path(image_path)
        vectors = encode_images(
            [path], self.clip, self.clip_processor, self.clip.device,
            batch_size=1,
        )
        scores, _ = classify_embeddings(
            vectors, labels, self.clip, self.clip_processor,
            self.clip.device, template,
        )
        metadata = pd.DataFrame([{
            "row_index": 0,
            "filename": path.name,
            "relative_path": str(path),
        }])
        return rank_labels(scores, labels, metadata, top_k)

    def ask(self, image_path, question, max_new_tokens=128):
        return answer_image(
            image_path, question, self.vlm, self.vlm_processor,
            max_new_tokens=max_new_tokens,
        )

    def caption(self, image_path):
        return self.ask(
            image_path,
            "Write one concise sentence describing the main visible subject "
            "and its surroundings. Describe only visible details. "
            "Do not guess identities, locations, or events.",
            max_new_tokens=96,
        )

    def locate(self, image_path, target):
        return locate_object(
            image_path, target, self.vlm, self.vlm_processor
        )
