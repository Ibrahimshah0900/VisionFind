"""Voice callback checks without ASR or browser; run chat suite for real components."""
import ast
import copy
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
import uuid
from unittest.mock import patch

root = Path(__file__).resolve().parents[1]
tree = ast.parse((root / 'src/multi_chat.py').read_text())
build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='build_multi_chat')
callback = next(n for n in build.body if isinstance(n, ast.FunctionDef) and n.name=='transcribe_voice')
voice_functions = runpy.run_path(str(root / 'src/voice.py'))
voice = types.ModuleType('src.voice')
voice.MAX_BYTES=20*1024*1024
voice.merge_transcript=voice_functions['merge_transcript']
voice.transcribe_audio=lambda path: {'text':'Count bowls.'}
package=types.ModuleType('src'); package.__path__=[]; package.voice=voice
class Gr:
    update = staticmethod(lambda **kwargs: kwargs)
    Info = staticmethod(lambda message: None)
    Warning = staticmethod(lambda message: None)
with tempfile.TemporaryDirectory() as tmp:
    folder=Path(tmp); audio=folder/'audio.wav'; audio.write_bytes(b'mock audio')
    ns={'__package__':'src','copy':copy,'gr':Gr,'root':folder,'uuid':uuid,'Path':Path,
        'save_json':lambda p,v:p.write_text(json.dumps(v))}
    exec(compile(ast.Module(body=[callback], type_ignores=[]),'voice_callback','exec'),ns)
    handler=ns['transcribe_voice']
    payload={'text':'Describe image 1.', 'files':['pending.png']}
    with patch.dict(sys.modules, {'src':package,'src.voice':voice}):
        updates=list(handler(str(audio),payload))
        assert updates[0][0]['interactive'] is False
        final=updates[-1][0]
        assert final['interactive'] is True
        assert final['value']['files']==['pending.png']
        assert final['value']['text']=='Describe image 1.\nCount bowls.'
        assert payload['text']=='Describe image 1.'
        assert updates[-1][1]=={'value':None,'visible':False}
        assert list((folder/'results/voice_checks').glob('*/transcript.json'))
        def fail(path): raise ValueError('No speech')
        voice.transcribe_audio=fail
        failure=list(handler(str(audio),payload))[-1]
        assert failure[0]['value']==payload and failure[0]['interactive'] is True
        assert list((folder/'results/voice_checks').glob('*/error.json'))
        missing=list(handler(None,payload))[-1]
        assert missing[0]['value']==payload
assert 'visionfind_chat_handler' not in ast.unparse(callback)
print('Voice callback success, error restoration, attachments, logs and no auto-submit: PASSED')
