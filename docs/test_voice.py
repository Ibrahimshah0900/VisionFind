"""Voice validation/composer checks; no ASR download or accuracy claim."""
from pathlib import Path
import runpy
import numpy as np
n=runpy.run_path(str(Path(__file__).resolve().parents[1]/'src/voice.py'))
validate=n['validate_samples']; merge=n['merge_transcript']
for samples in [np.zeros(160), np.ones(16000*61), [float('nan')], [[.1,.2]], []]:
    try: validate(samples)
    except ValueError: pass
    else: raise AssertionError('Invalid audio accepted')
assert validate(np.full(16000,.1)).shape==(16000,)
payload={'text':'Describe image 1.', 'files':['photo.png']}
out=merge(payload,{'text':'Draw boxes around the dog.'})
assert out['text']=='Describe image 1.\nDraw boxes around the dog.'
assert out['files']==['photo.png'] and out is not payload
out['files'].append('new.png'); assert payload['files']==['photo.png']
try: merge(payload,{'text':' '})
except ValueError: pass
else: raise AssertionError('Empty transcript accepted')
assert payload['text']=='Describe image 1.'
print('Voice audio validation, editable transcript and attachment preservation: PASSED')
