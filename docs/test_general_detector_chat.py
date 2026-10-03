"""No model downloads: category routing and reply consistency checks."""
import ast
from pathlib import Path
import re
import runpy
import tempfile

root = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(root / 'src/general_detector.py'))
reply = module['chat_reply']
class Overlay:
    def save(self, path):
        Path(path).write_text('mock overlay')
def fake_detection(path, target):
    return {'kind':'bowl','threshold':.55,'detected_count':3,'detections':[{}, {}, {}]}, Overlay()
reply.__globals__['detect_category'] = fake_detection
class Engine:
    def ask(self, *args, **kwargs):
        raise RuntimeError('description unavailable')
with tempfile.TemporaryDirectory() as folder:
    text, media, detail = reply(Engine(), 'mock.png', 'count bowls', {'target':'bowls','locate':False}, folder)
    assert '3 bowl' in text and not media and detail['box_count']==3
    text, media, detail = reply(Engine(), 'mock.png', 'describe and box bowls', {'target':'bowls','locate':True}, folder)
    assert len(media)==1 and Path(media[0][0]).exists() and detail['box_count']==3
    assert 'description_error' in detail
source = root / 'src/multi_chat.py'
tree = ast.parse(source.read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='analyze_one')
ns = {'re':re, 'detector_kind':lambda *a:None, 'resolve_general_target':module['resolve_target'],
      'general_detector_reply':lambda *a:('detector',[],{'box_count':3})}
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
for locate, message in [(True,'box bowls'), (False,'how many bowls?')]:
    text, _, details=ns['analyze_one'](None, {'id':1,'path':'mock.png'}, message,
        {'mode':'answer','target':'bowls','locate':locate}, None, [])
    assert text=='detector' and details['detector_result']['box_count']==3
ns['detector_kind']=lambda *a:'person'
ns['detector_reply']=lambda *a:(['person preserved'],[],{})
assert ns['analyze_one'](None,{'id':1,'path':'mock.png'},'box people',{'mode':'answer','target':'people','locate':True},None,[])[0]=='person preserved'
print('General detector chat routing, count/overlay, partial-description failure and person precedence: PASSED')
