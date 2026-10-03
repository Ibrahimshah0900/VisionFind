"""Short-video sampled-frame observations; no tracking or audio interpretation."""
from pathlib import Path
import json
import numpy as np

MAX_BYTES = 100 * 1024 * 1024
MAX_SECONDS = 30


def sample_video(video_path, folder, frame_count=8):
    import av
    path, folder = Path(video_path), Path(folder)
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError('Provide an existing video of at most 100 MiB.')
    if type(frame_count) is not int or not 2 <= frame_count <= 8:
        raise ValueError('Sample between two and eight frames.')
    folder.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode='r') as container:
        if not container.streams.video:
            raise ValueError('File has no video stream.')
        stream = container.streams.video[0]
        duration = (float(stream.duration * stream.time_base) if stream.duration is not None
                    else float(container.duration / av.time_base) if container.duration else None)
        if duration is None or not np.isfinite(duration) or not 0 < duration <= MAX_SECONDS:
            raise ValueError('Use a clip with known duration between 0 and 30 seconds.')
        rate = float(stream.average_rate) if stream.average_rate else 0
        end = max(0, duration - (1 / rate if rate > 0 else .1))
        targets = np.linspace(0, end, frame_count)
        records, next_target, first_pts, last_time = [], 0, None, -1
        for frame in container.decode(video=0):
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Video frame timestamps are missing.')
            absolute = float(frame.pts * frame.time_base)
            if first_pts is None:
                first_pts = absolute
            timestamp = absolute - first_pts
            if timestamp < last_time - 1e-6:
                raise ValueError('Decoded frame timestamps are not ordered.')
            last_time = timestamp
            if timestamp > MAX_SECONDS + .1:
                raise ValueError('Decoded video exceeds 30 seconds.')
            if next_target >= frame_count or timestamp + 1e-6 < targets[next_target]:
                continue
            if frame.width * frame.height > 4096 * 2160:
                raise ValueError('Video frame resolution exceeds the supported limit.')
            image = frame.to_image().convert('RGB')
            image.thumbnail((768, 768))
            saved = folder / f'frame_{len(records)+1:02d}.jpg'
            image.save(saved, quality=92)
            records.append({'timestamp_seconds': round(timestamp, 6), 'path': str(saved)})
            # One observation per actual frame; skip targets crossed by a gap.
            while next_target < frame_count and targets[next_target] <= timestamp + 1e-6:
                next_target += 1
        if not records:
            raise ValueError('No timestamped frames could be sampled.')
    metadata = {'duration_seconds': duration, 'frames': records,
                'requested_frame_count': frame_count, 'sampled_frame_count': len(records),
                'audio_analyzed': False, 'tracking': False,
                'limitations': 'Sparse visual samples can miss actions between frames. Timestamps are relative to the first decoded video frame.'}
    (folder / 'sampling.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def analyze_video(engine, video_path, output_folder, question='Explain what is visible in this video.'):
    folder = Path(output_folder)
    metadata = sample_video(video_path, folder / 'frames')
    report = {'question': question, 'sampling': metadata, 'observations': [],
              'accuracy_review': 'pending', 'weights_updated': False,
              'scope': 'Independent sampled-frame descriptions, not continuous video understanding.'}
    for i, record in enumerate(metadata['frames'], 1):
        print(f'Frame {i}/{len(metadata["frames"])} at {record["timestamp_seconds"]:.2f}s...', flush=True)
        answer = engine.ask(record['path'],
            'You have one sampled video frame. Describe only visible subjects and actions briefly. '
            'Do not infer motion direction, events between frames, sounds, identities or emotions. '
            'State uncertainty when necessary. Request: ' + question,
            max_new_tokens=110)
        report['observations'].append({**record, 'answer': answer})
        (folder / 'results.json').write_text(json.dumps(report, indent=2, default=str))
    return report
