"""Exercise the full probe on GPU0 with its actual native glue and portable WGMMA arithmetic."""
import argparse
import hashlib
import json
import linecache
import time
from pathlib import Path
import torch

H = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--ctx', type=int, default=4096)
ap.add_argument('--page', type=int, default=1)
ap.add_argument('--scale', type=float, default=1.)
a = ap.parse_args()
p = H / 'phases_lab.py'
s = p.read_text()
linecache.cache[str(p)] = (len(s), None, s.splitlines(True), str(p))
ns = {}
exec(compile(s, str(p), 'exec'), ns)
flush = torch.zeros(1 << 18, device='cuda', dtype=torch.int32)
ns['lab_cold_l2'] = lambda: flush.sum()
gen = ns['generate_input']
ns['generate_input'] = lambda **kw: gen(**dict(kw, ctx=a.ctx, page=a.page, kv_scale=a.scale))
rows = ns['lab'](time.monotonic() + 1500, {})
print('\n'.join(rows), flush=True)
assert sum(r.startswith('R ') for r in rows) == 4, rows
assert not any(r.startswith(('ERROR', 'SKIP')) for r in rows), rows
out = dict(passed=True, cases=4, ctx=a.ctx, page=a.page, scale=a.scale,
           task_gates=True, clocked_bitwise=True,
           payload_sha256=hashlib.sha256(s.encode()).hexdigest())
(H / f'local_{a.ctx}_p{a.page}_s{a.scale:g}.json').write_text(json.dumps(out, indent=2) + '\n')
