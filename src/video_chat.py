"""Short-video chat integration; image handlers remain independent."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
import uuid
import gradio as gr
from . import video, video_requests

EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}

def file_path(value):
    if isinstance(value, dict):
        return value.get('path') or value.get('name')
    return str(value)

def handles(payload, state):
    payload, state = payload or {}, state or {}
    files = payload.get('files') or []
    if any(Path(file_path(f)).suffix.lower() in EXTENSIONS for f in files):
        return True
    return bool(not files and state.get('video') and
                ('video' in (payload.get('text') or '').lower() or not state.get('images')))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str))

def encode(arguments):
    result = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                             *arguments], capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])

def render_people(path, folder, targets=None):
    targets = targets or ['person']
    import cv2
    from . import detectors, video_object_detector
    destination = folder / 'people_boxes_only.mp4'
    records = []
    with tempfile.TemporaryDirectory(prefix='visionfind_video_') as temp:
        temp = Path(temp)
        normalized, rendered, frame_path = temp/'input.mp4', temp/'boxes.mp4', temp/'frame.png'
        encode(['-i', str(path), '-an', '-vf', 'fps=10', '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p', str(normalized)])
        cap = cv2.VideoCapture(str(normalized))
        writer = None
        try:
            if not cap.isOpened():
                raise ValueError('Could not decode video.')
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width, height = (int(cap.get(prop)) for prop in
                             (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or width <= 0 or height <= 0 or total > 301:
                raise ValueError('Unexpected normalized video format.')
            writer = cv2.VideoWriter(str(rendered), cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError('Could not create video writer.')
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if len(records) >= 301:
                    raise ValueError('Video exceeds the supported frame limit.')
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError('Could not prepare detector input.')
                if targets == ['person']:
                    result, _ = detectors.detect_objects(str(frame_path), 'person')
                else:
                    result = video_object_detector.detect_video_objects(str(frame_path), targets)
                for detection in result['detections']:
                    x1, y1, x2, y2 = map(lambda v: int(round(v)), detection['bbox_original'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                writer.write(frame)
                records.append({'frame_index': len(records),
                    'timestamp_seconds': round(len(records)/fps, 6),
                    'detected_count': result['detected_count'], 'detections': result['detections']})
                if len(records) % 10 == 0:
                    print(f'Video: processed {len(records)}/{total} frames...', flush=True)
                    dump(folder/'frame_detections.json', records)
            if not records or len(records) != total:
                raise ValueError('Video decoding was incomplete.')
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            video_object_detector.release()
        encode(['-i', str(rendered), '-an', '-c:v', 'libx264', '-preset', 'veryfast',
                '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination)])
    dump(folder/'frame_detections.json', records)
    return destination, records

def respond(engine, payload, chat, state):
    payload = payload or {}
    chat, state = list(chat or []), copy.deepcopy(state or {})
    yield gr.update(interactive=False), chat, state
    folder = Path(engine.root)/'results/video_chat'/uuid.uuid4().hex
    folder.mkdir(parents=True, exist_ok=True)
    try:
        files = payload.get('files') or []
        if files:
            if len(files) != 1 or Path(file_path(files[0])).suffix.lower() not in EXTENSIONS:
                raise ValueError('Attach one video per request, without images in the same batch.')
            incoming = Path(file_path(files[0]))
            if not incoming.is_file() or incoming.stat().st_size > video.MAX_BYTES:
                raise ValueError('Use a video of at most 100 MiB and 30 seconds.')
            path = folder/('input'+incoming.suffix.lower())
            shutil.copyfile(incoming, path)
            video.sample_video(path, folder/'validation')
            state['video'] = {'path': str(path), 'sha256': digest(path)}
        if not state.get('video'):
            raise ValueError('Attach a video first.')
        path = Path(state['video']['path'])
        query = (payload.get('text') or 'Explain this video').strip()
        if len(query) > 2000:
            raise ValueError('Keep messages below 2,000 characters.')
        plan = video_requests.plan_video_request(query)
        chat.append({'role': 'user', 'content': query})
        dump(folder/'request.json', {'query': query, 'plan': plan, 'video': state['video']})
        if plan['clarification']:
            chat.append({'role': 'assistant', 'content': plan['clarification']})
        else:
            cache_path = Path(engine.root)/'results/video_chat_cache'/f"{cache_key(state['video']['sha256'], plan)}.json"
            cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
            # Reuse only results for this exact file and exact explanation request.
            if plan['explain']:
                report_path = cache.get('description_report') if query == cache.get('description_query') else None
                if report_path and Path(report_path).is_file():
                    report = json.loads(Path(report_path).read_text())
                else:
                    report = video.analyze_video(engine, path, folder/'description', query)
                    cache['description_report'] = str(folder/'description/results.json')
                    cache['description_query'] = query
                lines = ['**Sampled-frame observations — accuracy unverified.**']
                for observation in report['observations']:
                    answer = observation['answer']
                    answer = answer.get('answer', str(answer)) if isinstance(answer, dict) else str(answer)
                    lines.append(f"**{observation['timestamp_seconds']:.2f}s:** {answer}")
                lines.append('Sparse frames can miss events; audio is not analyzed.')
                chat.append({'role': 'assistant', 'content': '\n\n'.join(lines)})
            if plan['draw_boxes'] or plan['count_per_frame'] or plan.get('evidence_summary'):
                boxed = Path(cache.get('boxed_video', folder/'people_boxes_only.mp4'))
                data_path = Path(cache.get('detections', folder/'frame_detections.json'))
                if data_path.is_file() and (not plan['draw_boxes'] or boxed.is_file()):
                    detections = json.loads(data_path.read_text())
                else:
                    boxed, detections = render_people(path, folder, plan.get('targets') or ['person'])
                    data_path = folder/'frame_detections.json'
                cache.update(boxed_video=str(boxed), detections=str(data_path))
                counts = [record['detected_count'] for record in detections]
                if plan.get('evidence_summary'):
                    from . import video_evidence
                    evidence = video_evidence.summarize(detections, plan.get('targets') or ['person'])
                    dump(folder/'detection_evidence.json', evidence)
                    chat.append({'role':'assistant', 'content':video_evidence.render(evidence)})
                if plan['count_per_frame'] and not plan.get('evidence_summary'):
                    chat.append({'role': 'assistant', 'content':
                        f"Detected {', '.join(plan.get('targets') or ['person'])} regions per frame ranged from {min(counts)} to {max(counts)}. "
                        'This is not a unique-person total or a guaranteed count.'})
                if plan['draw_boxes']:
                    chat.append({'role': 'assistant', 'content':
                        f"{', '.join(plan.get('targets') or ['person']).capitalize()} boxes update during processed playback, without IDs. "
                        'Detections may miss or duplicate objects; processing is not verified as live real time.'})
                    chat.append({'role': 'assistant', 'content': gr.Video(value=str(boxed), show_label=False)})
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            dump(cache_path, cache)
        dump(folder/'status.json', {'plan': plan, 'execution': 'completed',
             'accuracy': 'review pending', 'state': state})
    except Exception as error:
        dump(folder/'error.json', {'error': f'{type(error).__name__}: {error}'})
        chat.append({'role': 'assistant', 'content': f'Video request could not complete: {error}'})
        yield gr.update(value=payload, interactive=True), chat, state
        return
    yield gr.update(value={'text': '', 'files': []}, interactive=True), chat, state


def cache_key(video_hash, plan):
    targets = sorted(set(plan.get('targets') or ['person']))
    # Preserve existing person caches; other targets use separate caches.
    return video_hash if targets == ['person'] else video_hash+'_resnet50v2_coco_055_v1_'+'_'.join(targets)
