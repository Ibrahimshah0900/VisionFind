"""Timestamp/sampling checks using synthetic decoded frames; no model download."""
from pathlib import Path
import runpy
import sys
import tempfile
import types
from unittest.mock import patch
from PIL import Image

root=Path(__file__).resolve().parents[1]
module=runpy.run_path(str(root/'src/video.py'))
class Frame:
    time_base=1; width=64; height=64
    def __init__(self, pts): self.pts=pts
    def to_image(self): return Image.new('RGB',(64,64),'blue')
class Container:
    duration=5000000
    def __init__(self):
        self.stream=types.SimpleNamespace(duration=5,time_base=1,average_rate=2)
        self.streams=types.SimpleNamespace(video=[self.stream])
        self.frames=[Frame(t) for t in [10,10.5,12.2,14.5]]
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def decode(self,**kwargs): return iter(self.frames)
class Engine:
    def __init__(self): self.paths=[]
    def ask(self,path,*args,**kwargs):
        self.paths.append(path);assert Path(path).is_file()
        return {'answer':'visible frame'}
with tempfile.TemporaryDirectory() as tmp:
    folder=Path(tmp);source=folder/'clip.mp4';source.write_bytes(b'mock')
    container=Container()
    av=types.ModuleType('av'); av.time_base=1000000;av.open=lambda *a,**k:container
    with patch.dict(sys.modules,{'av':av}):
        sampled=module['sample_video'](source,folder/'sample',8)
        assert sampled['sampled_frame_count']==3
        assert [r['timestamp_seconds'] for r in sampled['frames']]==[0,2.2,4.5]
        assert len({r['path'] for r in sampled['frames']})==3
        engine=Engine();report=module['analyze_video'](engine,source,folder/'answers')
        assert len(engine.paths)==3 and len(report['observations'])==3
        assert (folder/'answers/results.json').exists()
        container.stream.duration=31
        try: module['sample_video'](source,folder/'long')
        except ValueError: pass
        else: raise AssertionError('Long clip accepted')
        container.stream.duration=5;container.frames=[Frame(None)]
        try: module['sample_video'](source,folder/'missing_pts')
        except ValueError: pass
        else: raise AssertionError('Missing timestamps accepted')
print('Video sampling, relative timestamps, sparse gaps, saved-original frame calls and rejection checks: PASSED')
