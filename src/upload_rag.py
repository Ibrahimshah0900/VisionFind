"""Isolated retrieval + visual answers over a conversation's saved uploads.

This module does not change the existing collection index, chat handler, or UI.
It indexes originals with the already-loaded CLIP and grounds answers in actual
retrieved images. Ranked matches are not evidence that the query object exists.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path

import numpy as np


class UploadedImageRAG:
    SCHEMA_VERSION = 1

    def __init__(self, engine, items, cache_path):
        self.engine = engine
        self.items = items
        self.cache_path = Path(cache_path)
        self.vectors = None
        self.cache_status = None
        self.cache_reason = None
        config = engine.clip.config
        self.model = {
            "name": getattr(config, "_name_or_path", "openai/clip-vit-base-patch32"),
            "revision": getattr(config, "_commit_hash", None),
            "dimensions": int(getattr(config, "projection_dim", 512)),
        }
        processor = getattr(engine.clip_processor, "image_processor", None)
        self.model["image_processor"] = processor.to_dict() if processor else {}
        self.manifest = {"schema_version": self.SCHEMA_VERSION,
                         "model": self.model, "items": self.items}
        self.fingerprint = self._fingerprint(self.manifest)

    @staticmethod
    def _fingerprint(value):
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _file_hash(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @classmethod
    def from_state(cls, engine, state):
        """Load a matching persisted index, or build one from saved originals."""
        session_id = state.get("session_id", "")
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id):
            raise ValueError("A valid conversation session_id is required.")
        source_items = state.get("images", [])
        if not isinstance(source_items, list) or not 1 <= len(source_items) <= 10:
            raise ValueError("Index 1–10 saved conversation images.")
        root = Path(engine.root).resolve()
        items, seen = [], set()
        for entry in source_items:
            number = entry.get("id")
            if type(number) is not int or number < 1 or number in seen:
                raise ValueError("Image IDs must be unique positive integers.")
            seen.add(number)
            path = Path(entry["path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Saved image is missing or outside the project: {path}")
            items.append({"image_id": number, "path": str(path),
                          "filename": entry.get("filename", path.name),
                          "content_sha256": cls._file_hash(path)})
        index = cls(engine, items, root / "results/upload_rag" / session_id / "index.npz")
        index._load_or_build()
        return index

    def _validate_vectors(self, value):
        vectors = np.asarray(value, dtype=np.float32)
        expected = (len(self.items), self.model["dimensions"])
        if vectors.shape != expected or not np.isfinite(vectors).all():
            raise ValueError(f"Expected finite embeddings of shape {expected}.")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
            raise ValueError("Index embeddings must have unit lengths.")
        return vectors

    def _embed_images(self, paths):
        from .embeddings import encode_images
        return encode_images(paths, self.engine.clip, self.engine.clip_processor,
                             self.engine.clip.device, batch_size=4)

    def _embed_query(self, query):
        from .embeddings import encode_query
        return encode_query(query, self.engine.clip, self.engine.clip_processor,
                            self.engine.clip.device)

    def _load_or_build(self):
        if self.cache_path.exists():
            try:
                with np.load(self.cache_path, allow_pickle=False) as saved:
                    if str(saved["fingerprint"].item()) != self.fingerprint:
                        raise ValueError("Image set, content, or model settings changed.")
                    manifest = json.loads(str(saved["manifest"].item()))
                    if self._fingerprint(manifest) != self.fingerprint:
                        raise ValueError("Cache metadata does not match its fingerprint.")
                    self.vectors = self._validate_vectors(saved["vectors"])
                self.cache_status = "loaded"
                return
            except (ValueError, KeyError, OSError, EOFError) as error:
                self.cache_reason = str(error)
        self.vectors = self._validate_vectors(self._embed_images([x["path"] for x in self.items]))
        self._save()
        self.cache_status = "built"

    def _save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.cache_path.with_name(f".{uuid.uuid4().hex}.tmp")
        try:
            with temp.open("wb") as handle:
                np.savez_compressed(handle, vectors=self.vectors,
                    fingerprint=np.array(self.fingerprint),
                    manifest=np.array(json.dumps(self.manifest, sort_keys=True, default=str)))
            temp.replace(self.cache_path)
        finally:
            temp.unlink(missing_ok=True)

    def _verify_sources(self):
        for item in self.items:
            path = Path(item["path"])
            if not path.is_file() or self._file_hash(path) != item["content_sha256"]:
                raise RuntimeError("Indexed images changed or disappeared. Rebuild from the current state.")

    def search(self, query, top_k=3):
        """Return stable image IDs ranked by CLIP cosine similarity."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Provide a non-empty retrieval query.")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        self._verify_sources()
        vector = np.asarray(self._embed_query(query.strip()), dtype=np.float32)
        if vector.shape != (self.model["dimensions"],) or not np.isfinite(vector).all():
            raise ValueError("Query embedding has invalid shape or values.")
        length = np.linalg.norm(vector)
        if length == 0:
            raise ValueError("Query embedding cannot be zero.")
        scores = self.vectors @ (vector / length)
        positions = np.argsort(-scores, kind="stable")[:min(top_k, len(self.items))]
        return [{"rank": rank, "image_id": self.items[int(pos)]["image_id"],
                 "path": self.items[int(pos)]["path"],
                 "filename": self.items[int(pos)]["filename"],
                 "similarity": float(scores[int(pos)])}
                for rank, pos in enumerate(positions, 1)]

    def answer(self, search_query, question=None, top_k=1):
        """Retrieve first, then answer separately from each actual source image.

        Source IDs are attached by code; the VLM does not invent citations.
        This does not verify the answer or establish object absence/presence.
        """
        if question is None:
            question = search_query
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Provide a non-empty visual question.")
        matches = self.search(search_query, top_k)
        answers = []
        for match in matches:
            response = self.engine.ask(match["path"],
                "You have exactly one retrieved image. Answer from visible evidence in this image only. "
                "Retrieval rank does not prove an object is present. If the question's premise is not "
                "visible, say that it is not confirmed in this image. State uncertainty. "
                "Do not invent other images, claim to draw boxes, or guess health, emotions, identities "
                "or exact animal breeds.\nQuestion: " + question.strip(), max_new_tokens=140)
            answers.append({"image_id": match["image_id"], "source_path": match["path"],
                            "rank": match["rank"], "similarity": match["similarity"],
                            "answer": response["answer"]})
        return {"search_query": search_query, "question": question,
                "matches": matches, "answers": answers,
                "index_fingerprint": self.fingerprint,
                "note": "Matches are ranked candidates, not verified presence. Cosine scores are not probabilities."}
