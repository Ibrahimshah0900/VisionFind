"""Portable model-cache location for VisionFind."""
import os
from pathlib import Path

MODEL_CACHE_ROOT = Path(
    os.environ.get(
        "VISIONFIND_MODEL_CACHE",
        str(Path.home() / ".cache" / "visionfind")
    )
).expanduser()
