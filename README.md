# VisionFind

A conversational computer vision project for image search, visual questions,
object detection, and processed video annotations.

Built with CLIP, Qwen2.5-VL, PyTorch, Torchvision, Gradio, and FastAPI.
A local desktop chat interface connects to the GPU backend through a Python proxy.

## Features

- Describe and compare uploaded images.
- Answer visual questions and search uploaded images using CLIP.
- Attach up to ten images per conversation.
- Detect people, faces, and supported objects with annotated results.
- Process short videos with object boxes, detection counts, and timestamps.
- Review and edit voice transcripts before sending.
- Run authenticated API requests through a queued inference worker.

The dedicated image-object adapter supports bed, bowl, dog, truck,
dining table, and cup, alongside person and face detection.
Named video objects use supported COCO categories.

## Architecture

Local browser interface -> local Python proxy -> GPU FastAPI backend.

The backend routes requests to image retrieval, visual-language inference,
image detection, or video processing. Generated media is delivered through
authenticated asset endpoints.

## Backend setup

Use a CUDA GPU environment with FFmpeg available.
Install a compatible CUDA PyTorch/Torchvision pair separately.

Then run:

    python -m pip install -r requirements-backend.txt
    python run_backend.py

The launcher requests an API token through a hidden prompt.
Use a random token of at least 32 characters and configure the local
interface with the same value. Alternatively, set VISIONFIND_API_TOKEN
in the backend environment.

The API binds to 127.0.0.1:8000. Connecting a desktop interface to a
Kaggle or Colab backend requires an HTTPS tunnel to that API.
The API requires authentication. Keep the GPU runtime active.

The first launch downloads pretrained weights. Later launches reuse
the cache. Set VISIONFIND_MODEL_CACHE to choose the cache location.

CLIP runs on CPU. Qwen2.5-VL-3B runs on GPU in FP16.
Exact CLIP and Qwen revisions are recorded in model_config.json.
Detector and speech weights download separately when needed.

## Local interface

The desktop interface is included under frontend/.
Follow frontend/README.md for Windows setup.

The local Python proxy keeps the backend token out of browser code.
Its microphone uses browser speech recognition with editable text.
The backend Whisper implementation is separate and is not connected
to the local interface microphone.

## Example requests

- Describe image 1 and compare it with image 2.
- Count the people and draw boxes around them.
- Find the uploaded image showing a cup of coffee.
- Draw boxes around cars in this video.
- When were cars detected in this video?

## Validation

Eleven existing regression scripts passed on the reconstructed staging
source before the portable cache-path changes. These checks primarily
use fixtures or mocked inference; they are not a general accuracy benchmark.

This source combines the transfer backup with later fixes recovered
from the saved Kaggle notebook. It has not been compared byte-for-byte
with the final saved Kaggle output.

Fresh model startup through run_backend.py remains unverified.

## Known limitations

- Visual descriptions can omit details or hallucinate.
- Video scene and movement explanations remain experimental.
- Detection counts are predictions rather than guaranteed totals.
- Annotated video playback is processed output, not live inference.
- Videos are limited to 30 seconds and 100 MiB.
- Backend sessions and jobs disappear after a runtime restart.
- Camera capture is not implemented.
- No fine-tuning was performed.

## Project structure

- src/: reusable inference and chat modules.
- run_backend.py: portable GPU backend launcher.
- model_config.json: recorded model revisions.
- requirements-backend.txt: backend dependencies.
- data/index/: six-image demonstration index.
- data/sample_images/: images referenced by the index.
- docs/: existing tests, development notes, and evaluation scripts.
- frontend/: local desktop interface.

Model weights, credentials, private recordings, user uploads,
large evaluation datasets, and generated results are excluded.

Development notes contain historical and environment-specific instructions.
Use this README and run_backend.py as the release entry point.

## GPU notebook

Open notebooks/gpu_backend.ipynb in Kaggle or Colab for the clean setup and startup cells. An HTTPS tunnel must be configured separately for desktop access. Fresh GPU startup remains unverified.
