"""Tests without using a real token, tunnel or GPU."""
import os
for name in ('ALL_PROXY', 'HTTPS_PROXY', 'HTTP_PROXY', 'all_proxy', 'https_proxy', 'http_proxy'):
    os.environ.pop(name, None)
os.environ['VISIONFIND_API_URL']='https://example.invalid'
os.environ['VISIONFIND_API_TOKEN']='test-token-not-a-real-secret'
import httpx
from fastapi.testclient import TestClient
from app import app

class MediaStream(httpx.AsyncByteStream):
    async def __aiter__(self): yield b'abc'

def fake(request):
    assert request.headers['authorization']=='Bearer test-token-not-a-real-secret'
    if request.url.path=='/health': return httpx.Response(200,json={'status':'ready'})
    if request.url.path=='/sessions': return httpx.Response(200,json={'session_id':'s'*24})
    if request.url.path.endswith('/uploads'):
        assert b'coffee.png' in request.content
        return httpx.Response(200,json={'upload_ids':['u'*24]})
    if request.url.path.startswith('/assets/'):
        if request.headers.get('range'):
            return httpx.Response(206,stream=MediaStream(),headers={'Content-Type':'video/mp4','Content-Range':'bytes 0-2/10'})
        return httpx.Response(200,content=b'image',headers={'Content-Type':'image/png'})
    return httpx.Response(404,json={'detail':'missing'})

with TestClient(app) as client:
    import asyncio
    original=app.state.client
    app.state.client=httpx.AsyncClient(base_url='https://example.invalid',transport=httpx.MockTransport(fake),headers={'Authorization':'Bearer test-token-not-a-real-secret'})
    assert client.get('/').status_code==200
    assert client.get('/api/connection').json()['connected']
    assert 'test-token' not in client.get('/').text
    assert client.post('/api/sessions',json={}).json()['session_id']=='s'*24
    assert client.post('/api/sessions',headers={'Origin':'https://evil.invalid'},json={}).status_code==403
    assert client.get('/api/not-allowed').status_code==404
    response=client.post('/api/sessions/'+'s'*24+'/uploads',files={'files':('coffee.png',b'fake image','image/png')})
    assert response.json()['upload_ids']==['u'*24]
    response=client.get('/api/assets/'+'a'*24,headers={'Range':'bytes=0-2'})
    assert response.status_code==206 and response.content==b'abc'
    assert response.headers['content-range']=='bytes 0-2/10'
    client.portal.call(app.state.client.aclose)
    app.state.client=original
print('LOCAL PROXY / AUTH / UPLOAD / MEDIA RANGE TESTS: PASSED')
