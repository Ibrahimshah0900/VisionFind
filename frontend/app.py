"""Local UI proxy: the Kaggle token stays on this Python server."""
from pathlib import Path
from contextlib import asynccontextmanager
import os
import re
import json
from urllib.parse import urlparse
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.background import BackgroundTask

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / '.env')
REMOTE = os.environ.get('VISIONFIND_API_URL', '').rstrip('/')
TOKEN = os.environ.get('VISIONFIND_API_TOKEN', '')

@asynccontextmanager
async def lifespan(app):
    if not REMOTE.startswith('https://') or not TOKEN:
        raise RuntimeError('Run python configure.py first to set your HTTPS URL and API token.')
    async with httpx.AsyncClient(base_url=REMOTE, headers={'Authorization': 'Bearer ' + TOKEN},
                                 timeout=httpx.Timeout(60, connect=15), follow_redirects=False) as client:
        app.state.client = client
        yield

app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])

@app.middleware('http')
async def local_origin(request, call_next):
    origin = request.headers.get('origin')
    if origin and origin not in {'http://127.0.0.1:7860', 'http://localhost:7860'}:
        return JSONResponse({'detail': 'Use the local VisionFind page.'}, status_code=403)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response

@app.get('/')
async def index():
    return FileResponse(BASE / 'static' / 'index.html')

@app.get('/api/connection')
async def connection(request: Request):
    try:
        response = await request.app.state.client.get('/health')
        if response.status_code != 200:
            return JSONResponse({'connected': False, 'detail': 'Backend rejected the connection. Check URL and token.'}, status_code=503)
        return {'connected': True}
    except httpx.HTTPError:
        return JSONResponse({'connected': False, 'detail': 'Kaggle is unreachable. Keep its API and tunnel running.'}, status_code=503)

@app.api_route('/api/{path:path}', methods=['GET', 'POST', 'DELETE'])
async def proxy(path: str, request: Request):
    ident = r'[A-Za-z0-9_-]{16,80}'
    permitted = {'GET': [rf'jobs/{ident}', rf'assets/{ident}'],
                 'POST': ['sessions', rf'sessions/{ident}/uploads', rf'sessions/{ident}/turns'],
                 'DELETE': [rf'sessions/{ident}']}
    if not any(re.fullmatch(pattern, path) for pattern in permitted[request.method]):
        raise HTTPException(404, 'Unknown API route')
    client = request.app.state.client
    try:
        if request.method == 'GET' and path.startswith('assets/'):
            headers = {}
            if 'range' in request.headers:
                headers['Range'] = request.headers['range']
            upstream = await client.send(client.build_request('GET', '/' + path, headers=headers), stream=True)
            if upstream.status_code not in (200, 206):
                await upstream.aclose()
                raise HTTPException(upstream.status_code, 'Media is unavailable. The Kaggle session may have restarted.')
            headers = {k: v for k, v in upstream.headers.items() if k in
                       ('content-type', 'content-length', 'content-range', 'accept-ranges')}
            return StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code,
                                     headers=headers, background=BackgroundTask(upstream.aclose))
        if path.endswith('/uploads'):
            form = await request.form()
            files = form.getlist('files')
            if not 1 <= len(files) <= 10:
                await form.close()
                raise HTTPException(400, 'Attach up to 10 images or one video.')
            if any(not hasattr(f, 'file') for f in files):
                await form.close()
                raise HTTPException(400, 'Invalid upload.')
            if sum(f.size or 0 for f in files) > 200 * 1024**2:
                await form.close()
                raise HTTPException(413, 'Attachments are too large.')
            try:
                response = await client.post('/' + path, files=[
                    ('files', (f.filename, f.file, f.content_type)) for f in files])
            finally:
                await form.close()
        else:
            body = None
            if request.method == 'POST':
                raw = await request.body()
                if len(raw) > 64000:
                    raise HTTPException(413, 'Message is too large.')
                body = json.loads(raw) if raw else {}
            response = await client.request(request.method, '/' + path, json=body)
        try:
            payload = response.json()
        except ValueError:
            payload = {'detail': 'Backend is unavailable. Check the Kaggle tunnel.'}
        return JSONResponse(payload, status_code=response.status_code)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(400, 'Invalid message.')
    except httpx.HTTPError:
        raise HTTPException(502, 'Connection interrupted. Check Kaggle and the tunnel, then reload to resume polling.')

app.mount('/static', StaticFiles(directory=BASE / 'static'), name='static')
