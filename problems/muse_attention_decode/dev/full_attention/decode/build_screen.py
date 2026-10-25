"""One attention submission for normal glue and every forced native decode arm."""
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4]
base=R/'runs/muse_attention_decode/20261025T030917Z_submit_81cccbc19410433b.py'
s=base.read_text();assert hashlib.sha256(s.encode()).hexdigest().startswith('2f53b672')
rows=json.loads((H/'manifest.json').read_text())['variants'];sources={r['name']:(H/(r['name']+'.py')).read_text() for r in rows}
s+='\n\n_DECODE_SOURCES='+repr(sources)+'''
_DECODE_MODULES={}
_DECODE_ARM='ref'
_DECODE_FORCE=False
_DECODE_BASE_FN=kernel_fn
def _decode_module(name):
    if name not in _DECODE_MODULES:
        body=_DECODE_SOURCES[name];mod={}
        exec(compile(body,'<decode_'+name+'>','exec'),mod)
        _DECODE_MODULES[name]=mod
    return _DECODE_MODULES[name]

def kernel_fn(data):
    cfg=data['config']
    if cfg['q_len']==1 and not cfg['window'] and cfg['batch'] in (16,32) and cfg['num_kv_heads']==2:
        if _DECODE_FORCE or torch.cuda.get_device_capability()[0]==9:
            return _decode_module(_DECODE_ARM)['attention'](data)
    return _DECODE_BASE_FN(data)
'''
(H/'board.py').write_text(s);print('Local screen board',hashlib.sha256(s.encode()).hexdigest())
