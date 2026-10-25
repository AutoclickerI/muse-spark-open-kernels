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
a=ap.parse_args()
ps = [H / p for p in a.payload]
# Pin the host as well as the measured S13 kernels, so a later board improvement
# does not silently change this experiment's wrapper or admission hash.
base='81cccbc19410433b'
row=next(json.loads(x) for x in (R/'runs/ledger.jsonl').read_text().splitlines() if json.loads(x).get('job')==base)
embedded=[h200_lab.embed(p) for p in ps]
s=(R/row['source']).read_text().rstrip()+'\n\n'+h200_lab.WRAPPER.format(
    names=', '.join(f'{n} {sha}' for n,sha,_ in embedded),base=base,payloads=embedded,
    budget=300-h200_lab.MARGIN_S,limit=h200_lab.LIMIT)
assert s.count('def prepare(weights):') == 1
s = s.replace('def prepare(weights):', 'def prepare(weights=None):')
assert s.count('state = _lab_base_prepare(weights)') == 1
s = s.replace('state = _lab_base_prepare(weights)', 'state = _lab_base_prepare()')
s += '\n# Full-attention diagnostic boot: attention has no harness prepare hook.\nif torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9:\n    prepare()\n'
out = H / ('_'.join(p.stem.removesuffix('_lab') for p in ps)+'_board.py')
out.write_text(s)
print(out, 'base', base, hashlib.sha256(s.encode()).hexdigest())
