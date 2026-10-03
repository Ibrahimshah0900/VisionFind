"""CLIP model loading utilities for VisionFind."""

import torch
from transformers import CLIPModel, CLIPProcessor


def load_clip(model_name: str, device: torch.device):
    """Load a CLIP model and its matching processor for inference."""
    processor = CLIPProcessor.from_pretrained(
        model_name,
        use_fast=False,
    )

    model = CLIPModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    return model, processor
