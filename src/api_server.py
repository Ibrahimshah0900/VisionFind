"""VisionFind authenticated demo API. One inference worker; no model changes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import RLock, BoundedSemaphore
from typing import List
import secrets
import time
import logging
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

class Turn(BaseModel):
    text: str = Field(default='', max_length=12000)
    upload_ids: List[str] = Field(default_factory=list, max_length=10)


def create_api(handler, root, token):
    if len(token) < 32:
        raise ValueError('Use a random API token of at least 32 characters.')
    root = Path(root).resolve()
    storage = root / 'results' / 'api_uploads'
    storage.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title='VisionFind API', docs_url=None, redoc_url=None, openapi_url=None)
    bearer = HTTPBearer(auto_error=False)
    sessions, jobs, uploads, assets = {}, {}, {}, {}
    lock = RLock()
    slots = BoundedSemaphore(10)
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='visionfind-inference')

    def auth(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
        if not credentials or not secrets.compare_digest(credentials.credentials, token):
            raise HTTPException(401, 'Invalid API token')

    def get_session(sid):
        if sid not in sessions:
            raise HTTPException(404, 'Session not found')
        return sessions[sid]

    def asset(path, kind):
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError('Output media file is missing')
        if not any(path.is_relative_to(root / folder) for folder in ('results', 'data')):
            import os
            import shutil
            import tempfile
            cache = Path(
                os.environ.get(
                    'GRADIO_TEMP_DIR',
                    str(Path(tempfile.gettempdir()) / 'gradio')
                )
            ).resolve()
            if not path.is_relative_to(cache):
                raise ValueError('Output file is outside approved media folders')
            destination_folder = root / 'results' / 'api_assets'
            destination_folder.mkdir(parents=True, exist_ok=True)
            destination = destination_folder / (
                secrets.token_urlsafe(24) + path.suffix
            )
            shutil.copyfile(path, destination)
            path = destination
        key = secrets.token_urlsafe(24)
        assets[key] = path
        return {'kind': kind, 'asset_id': key, 'url': '/assets/' + key}

    def content(value):
        if isinstance(value, str):
            return {'kind': 'text', 'text': value}
        kind = type(value).__name__.lower()
        if kind == 'gallery':
            items = []
            gallery = getattr(value, 'value', None) or []
            if hasattr(gallery, 'model_dump'):
                gallery = gallery.model_dump()
            if isinstance(gallery, dict):
                gallery = gallery.get('root', gallery.get('value', []))
            for item in gallery:
                if hasattr(item, 'model_dump'):
                    item = item.model_dump()
                caption = None
                if isinstance(item, (tuple, list)):
                    item, caption = item[0], item[1] if len(item) > 1 else None
                if isinstance(item, dict):
                    caption = item.get('caption', caption)
                    item = item.get('image', item.get('path'))
                    if hasattr(item, 'model_dump'):
                        item = item.model_dump()
                    if isinstance(item, dict):
                        item = item.get('path') or item.get('name')
                entry = asset(item, 'image')
                entry['caption'] = caption
                items.append(entry)
            return {'kind': 'gallery', 'items': items}
        if kind not in ('image', 'video', 'audio', 'file'):
            raise ValueError('Unsupported chat content: ' + kind)
        value = getattr(value, 'value', None)
        if isinstance(value, (tuple, list)):
            value = value[0]
        if isinstance(value, dict):
            value = value.get('path') or value.get('name') or value.get('video')
            if isinstance(value, dict):
                value = value.get('path')
        return asset(value, kind)

    def messages(chat):
        return [{'role': m['role'], 'content': content(m['content'])} for m in chat]

    @app.get('/health', dependencies=[Depends(auth)])
    def health():
        return {'status': 'ready', 'inference_workers': 1, 'max_pending_jobs': 10}

    @app.post('/sessions', dependencies=[Depends(auth)])
    def new_session():
        with lock:
            if len(sessions) >= 100:
                raise HTTPException(429, 'Session limit reached; restart API or delete old sessions')
            sid = secrets.token_urlsafe(24)
            sessions[sid] = {'chat': [], 'state': {}, 'busy': False}
        return {'session_id': sid}

    @app.delete('/sessions/{sid}', dependencies=[Depends(auth)])
    def delete_session(sid):
        with lock:
            session = get_session(sid)
            if session['busy']:
                raise HTTPException(409, 'Session has a pending job')
            del sessions[sid]
        return {'deleted': True}

    @app.post('/sessions/{sid}/uploads', dependencies=[Depends(auth)])
    async def upload(sid: str, files: List[UploadFile] = File(...)):
        with lock:
            get_session(sid)
        image_ext = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        video_ext = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
        extensions = [Path(f.filename or '').suffix.lower() for f in files]
        is_video = any(e in video_ext for e in extensions)
        if not files or len(files) > 10 or (is_video and len(files) != 1):
            raise HTTPException(400, 'Upload up to 10 images or one video')
        if any(e not in image_ext | video_ext for e in extensions):
            raise HTTPException(400, 'Unsupported file format')
        saved = []
        try:
            for f, ext in zip(files, extensions):
                uid = secrets.token_urlsafe(24)
                path = storage / (uid + ext)
                size = 0
                limit = 100 * 1024**2 if ext in video_ext else 20 * 1024**2
                saved.append((uid, path))
                with path.open('wb') as target:
                    while block := await f.read(1024 * 1024):
                        size += len(block)
                        if size > limit:
                            raise HTTPException(413, 'Upload exceeds file size limit')
                        target.write(block)
                if ext in image_ext:
                    from PIL import Image
                    with Image.open(path) as image:
                        image.verify()
            with lock:
                for uid, path in saved:
                    uploads[uid] = (sid, path)
            return {'upload_ids': [uid for uid, _ in saved]}
        except Exception as error:
            for _, path in saved:
                path.unlink(missing_ok=True)
            if isinstance(error, HTTPException):
                raise
            raise HTTPException(400, 'Invalid uploaded media') from error
        finally:
            for f in files:
                await f.close()

    def run_job(jid, sid, payload):
        try:
            with lock:
                job, session = jobs[jid], sessions[sid]
                job['status'] = 'running'
                chat, state = session['chat'], session['state']
            yielded = False
            for _, chat, state in handler(payload, chat, state):
                yielded = True
                with lock:
                    jobs[jid]['updates'] += 1
            if not yielded:
                raise RuntimeError('Chat handler returned no response')
            encoded = messages(chat)
            with lock:
                session.update(chat=chat, state=state)
                jobs[jid].update(status='completed', messages=encoded)
        except Exception:
            logging.exception('VisionFind API job failed: %s', jid)
            with lock:
                jobs[jid].update(status='failed', error='Request failed; inspect Colab logs.')
        finally:
            with lock:
                sessions[sid]['busy'] = False
            slots.release()

    @app.post('/sessions/{sid}/turns', status_code=202, dependencies=[Depends(auth)])
    def turn(sid: str, request: Turn):
        with lock:
            session = get_session(sid)
            if session['busy']:
                raise HTTPException(409, 'Wait for this session’s current job')
            paths = []
            for uid in request.upload_ids:
                if uid not in uploads or uploads[uid][0] != sid:
                    raise HTTPException(400, 'Upload does not belong to this session')
                paths.append(str(uploads[uid][1]))
            if not request.text.strip() and not paths:
                raise HTTPException(400, 'Send text or media')
            if len(jobs) >= 500:
                raise HTTPException(429, 'Demo job limit reached; restart API')
            if not slots.acquire(blocking=False):
                raise HTTPException(429, 'Inference queue is full')
            jid = secrets.token_urlsafe(24)
            jobs[jid] = {'job_id': jid, 'status': 'queued', 'updates': 0}
            session['busy'] = True
            worker.submit(run_job, jid, sid, {'text': request.text, 'files': paths})
        return {'job_id': jid, 'status_url': '/jobs/' + jid}

    @app.get('/jobs/{jid}', dependencies=[Depends(auth)])
    def job(jid: str):
        with lock:
            if jid not in jobs:
                raise HTTPException(404, 'Job not found')
            return dict(jobs[jid])

    @app.get('/assets/{key}', dependencies=[Depends(auth)])
    def get_asset(key: str):
        with lock:
            path = assets.get(key)
        if path is None or not path.is_file():
            raise HTTPException(404, 'Asset not found')
        return FileResponse(path)

    app.state.inference_worker = worker
    return app
