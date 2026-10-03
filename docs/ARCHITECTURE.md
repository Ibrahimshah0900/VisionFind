# VisionFind architecture

## Request flow

The desktop browser connects to the local FastAPI proxy in frontend/app.py.
The proxy forwards authenticated requests to the GPU backend and streams
returned media to the browser.

The backend API creates sessions, accepts uploads, queues chat requests,
and exposes job status and generated assets. A single inference worker
serializes access to the models.

## Main components

| Component | Implementation |
| --- | --- |
| Session chat and request handling | src/multi_chat.py |
| Query routing | src/query_router.py |
| CLIP collection search and classification | src/pipeline.py |
| Uploaded-image retrieval and visual answers | src/upload_rag.py |
| Visual questions | src/vqa.py |
| Approximate VLM localization | src/grounding.py |
| Person and face detection | src/detectors.py |
| Dedicated image-object detection | src/general_detector.py |
| Stronger pretrained detector | src/detector_candidate.py |
| Video requests and processing | src/video_chat.py |
| Detection-derived video summaries | src/video_evidence.py |
| Backend speech transcription | src/voice.py |
| Authenticated API | src/api_server.py |

## Models

CLIP uses openai/clip-vit-base-patch32.
Visual-language inference uses Qwen/Qwen2.5-VL-3B-Instruct.
Their recorded revisions are stored in model_config.json.

Person detection uses Torchvision Faster R-CNN MobileNet.
Named non-person video objects use Faster R-CNN ResNet-50 FPN v2.
Face detection uses OpenCV YuNet.
Backend speech transcription uses multilingual Whisper Base through
faster-whisper on CPU with INT8 computation.

These are pretrained models; no fine-tuning was performed.

## Storage and runtime

The included six-image CLIP index is stored in data/index/.
Its metadata references data/sample_images/ using relative paths.

Model downloads are cached outside the repository by default.
VISIONFIND_MODEL_CACHE overrides the cache location.

Generated outputs are written under results/.
API sessions and jobs live in runtime memory.

## Interface

The desktop interface contains Chat and Help tabs.
Image attachments appear as thumbnails; annotated results render larger.
Video output is processed playback with frame-dependent boxes.

Desktop microphone dictation uses browser speech recognition.
The separate backend Whisper module is not wired to that microphone.
