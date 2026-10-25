"""Attention has no weights/prepare hook: execute the synthetic lab during module initialization."""
import hashlib
import argparse
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
R = H.parents[3]
sys.path.insert(0, str(R / 'tools'))
import h200_lab

ap=argparse.ArgumentParser()
ap.add_argument('payload',nargs='*',default=['phases_lab.py'])
ap.add_argument('-o','--out')
a=ap.parse_args()
ps = [H / p for p in a.payload]
# Pin the host as well as the measured S13 kernels, so a later board improvement
# does not silently change this experiment's wrapper or admission hash.
base='81cccbc19410433b'
row=next(json.loads(x) for x in (R/'runs/ledger.jsonl').read_text().splitlines() if json.loads(x).get('job')==base)
embedded=[h200_lab.embed(p) for p in ps]
s=(R/row['source']).read_text().rstrip()+'\n\n'+h200_lab.WRAPPER.format(
    names=', '.join(f'{n} {sha}' for n,sha,_ in embedded),base=base,payloads=embedded,
    limit=h200_lab.LIMIT)
# This pinned attention source has a private zero-argument prepare tuner. Keep
# its ABI and the established guarded boot while sharing the current lab runner.
s+='''
_lab_base_prepare = prepare
def prepare(weights=None):
    import time as _t
    _start = _t.monotonic()
    state = _lab_base_prepare()
    _lab_run(weights, state, _start, _start + 285, "import")
    return state
'''
s += '\n# Full-attention diagnostic boot: attention has no harness prepare hook.\nif torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9:\n    prepare()\n'
out = H / (a.out or ('_'.join(p.stem.removesuffix('_lab') for p in ps)+'_board.py'))
out.write_text(s)
print(out, 'base', base, hashlib.sha256(s.encode()).hexdigest())
