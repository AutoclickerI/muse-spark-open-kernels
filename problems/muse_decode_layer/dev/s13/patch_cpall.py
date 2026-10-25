"""Replace the T=1 coded GEMV with c446fdab's all-thread staging and k-major bank."""
import hashlib
from pathlib import Path
BODY_SHA = '1c1efc81eac5cad536358025f40389019449cac518ac2c56e9d3b02ef2bd3916'

def patch(src):
    body = Path(__file__).with_name('cpall_frozen.py').read_text()
    assert hashlib.sha256(body.encode()).hexdigest() == BODY_SHA
    body = body.split('import torch.nn.functional as F\n', 1)[1]
    old = 'CD_INSTANCES = tuple((nt, rt, pipe, v) for v in (3, 4) for nt in (1, 2, 4) for rt in (1, 2) for pipe in (2, 4))'
    assert body.count(old)==1
    body=body.replace(old,'CD_INSTANCES = ((1, 1, 2, 3),)')
    a=src.index('\nCD_CUDA = ');b=src.index('\n# (tokens, role) ->',a)
    src=src[:a]+'\n'+body+'\n'+src[b:]
    old='        c = cd_encode(w)'
    assert src.count(old)==1
    src=src.replace(old,'        c = cd_encode(w, kmajor=True)')
    return src
