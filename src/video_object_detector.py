"""Video object adapter; existing person and image paths are independent."""
from . import detector_candidate

BACKEND_VERSION = 'resnet50v2_coco_055_v1'

def detect_video_objects(path, targets):
    return detector_candidate.detect(path, targets)

def release():
    detector_candidate.release()
