"""Speech-to-text only; never auto-submit or modify chat state."""
from src.config import MODEL_CACHE_ROOT
from pathlib import Path
import copy
import time
import numpy as np

SAMPLE_RATE = 16000
MAX_SECONDS = 60
MAX_BYTES = 20 * 1024 * 1024
CACHE = (MODEL_CACHE_ROOT / 'speech')
_model = None


def validate_samples(samples):
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
        raise ValueError('Audio must contain finite mono samples.')
    duration = samples.size / SAMPLE_RATE
    if duration > MAX_SECONDS:
        raise ValueError(f'Voice queries must be at most {MAX_SECONDS} seconds.')
    if float(np.max(np.abs(samples))) < 1e-5:
        raise ValueError('No audible input detected. Record a spoken query.')
    return samples


def load_speech_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        CACHE.mkdir(parents=True, exist_ok=True)
        _model = WhisperModel('base', device='cpu', compute_type='int8',
                             cpu_threads=4, download_root=str(CACHE))
    return _model


def decode_query_audio(path):
    """Decode bounded mono float32 audio without removed PyAV keywords."""
    import av
    chunks = []
    count = 0
    resampler = av.AudioResampler(format='flt', layout='mono', rate=SAMPLE_RATE)
    def retain(frames):
        nonlocal count
        for frame in frames:
            samples = np.asarray(frame.to_ndarray(), dtype=np.float32).reshape(-1)
            count += samples.size
            if count > SAMPLE_RATE * MAX_SECONDS:
                raise ValueError(f'Voice queries must be at most {MAX_SECONDS} seconds.')
            chunks.append(samples)
    with av.open(str(path), mode='r') as container:
        if not container.streams.audio:
            raise ValueError('The file has no audio stream.')
        for frame in container.decode(audio=0):
            retain(resampler.resample(frame))
        retain(resampler.resample(None))
    if not chunks:
        raise ValueError('No audio samples decoded.')
    return np.concatenate(chunks)


def transcribe_audio(audio_path, language=None):
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError('Recorded audio is missing.')
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('Audio exceeds the 20 MiB limit.')
    samples = validate_samples(decode_query_audio(path))
    model = load_speech_model()
    started = time.perf_counter()
    segments, info = model.transcribe(samples, language=language, task='transcribe',
        beam_size=1, temperature=0, condition_on_previous_text=False, vad_filter=True)
    rows = [{'start': float(s.start), 'end': float(s.end), 'text': s.text.strip()}
            for s in segments]
    text = ' '.join(row['text'] for row in rows if row['text']).strip()
    if not text:
        raise ValueError('No speech was transcribed. Try a clearer recording.')
    return {'text': text, 'segments': rows, 'language': info.language,
            'language_probability': float(info.language_probability),
            'duration_seconds': samples.size / SAMPLE_RATE,
            'transcription_seconds': round(time.perf_counter()-started, 3),
            'model': 'Systran/faster-whisper-base', 'device': 'cpu', 'compute_type': 'int8',
            'accuracy_review': 'pending'}


def merge_transcript(payload, transcript):
    """Return editable composer content, preserving pending image attachments."""
    if not isinstance(payload, dict):
        raise ValueError('Composer payload must be a dictionary.')
    text = transcript.get('text') if isinstance(transcript, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError('An editable nonempty transcript is required.')
    existing = payload.get('text', '')
    if not isinstance(existing, str):
        raise ValueError('Composer text must be a string.')
    result = copy.deepcopy(payload)
    result['text'] = '\n'.join(s for s in [existing.strip(), text.strip()] if s)
    return result
