from pathlib import Path
from tempfile import TemporaryDirectory
import time
from fastapi.testclient import TestClient
from src.api_server import create_api

def run():
    class Gallery:
        def __init__(self, value): self.value = value
    def fake(payload, chat, state):
        yield {}, chat + [{'role':'assistant', 'content':'Echo: ' + payload['text']},
                         {'role':'user','content':Gallery([(str(media), 'Image 1')])}], state
    with TemporaryDirectory() as folder:
        media = Path(folder) / 'data' / 'test.png'
        media.parent.mkdir()
        media.write_bytes(b'test asset')
        app = create_api(fake, Path(folder), 't' * 48)
        try:
            with TestClient(app) as client:
                assert client.get('/health').status_code == 401
                headers = {'Authorization':'Bearer ' + 't' * 48}
                assert client.get('/health',headers=headers).status_code == 200
                sid = client.post('/sessions',headers=headers).json()['session_id']
                bad = client.post(f'/sessions/{sid}/turns',headers=headers,json={'text':'hello','upload_ids':['unknown']})
                assert bad.status_code == 400
                reply = client.post(f'/sessions/{sid}/turns',headers=headers,json={'text':'hello'})
                assert reply.status_code == 202
                jid = reply.json()['job_id']
                for _ in range(100):
                    job = client.get('/jobs/' + jid,headers=headers).json()
                    if job['status'] in ('completed','failed'): break
                    time.sleep(.01)
                assert job['status'] == 'completed', job
                assert job['messages'][0]['content']['text'] == 'Echo: hello'
                gallery = job['messages'][1]['content']
                assert gallery['kind'] == 'gallery'
                assert gallery['items'][0]['caption'] == 'Image 1'
                assert client.get(gallery['items'][0]['url'],headers=headers).content == b'test asset'
                assert client.get('/assets/unknown',headers=headers).status_code == 404
                assert client.post('/sessions/'+sid+'/uploads',headers=headers,files={'files':('bad.png',b'not an image','image/png')}).status_code == 400
            print('API AUTH / JOB / UPLOAD / ASSET CHECKS: PASSED')
        finally:
            app.state.inference_worker.shutdown(wait=True)

if __name__ == '__main__': run()
