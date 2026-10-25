"""Preserve a compact structural-lab receipt, including CUPTI node evidence."""
import ast
import base64
import hashlib
import json
import lzma
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
job = sys.argv[1]
receipt, = (REPO / 'runs/muse_prefill_layer').glob(f'*_{job}.json')
wrapper = receipt.with_suffix('.py').read_text()
tree = ast.parse(wrapper)
payloads = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
    and any(getattr(t, 'id', '') == '_LAB_PAYLOADS' for t in n.targets)))
name, _, payload = payloads[0]
log, = (REPO / 'problems/muse_prefill_layer/lab/profiles').glob(f'*.{job}.log')
lines = log.read_text().splitlines()
result = dict(job=job, host='muse_prefill_layer', base_sha256='acda6c62',
    payload_name=name, payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),
    receipt=str(receipt.relative_to(REPO)), loads={}, pairs={}, increments={}, epilogue_us={})
for line in lines:
    if line.startswith('LOAD '):
        _, case, *values = line.split()
        result['loads'][case] = {k:float(v) for k,v in (s.split('=') for s in values)}
    elif line.startswith('PAIR '):
        _, case, *values = line.split()
        assert 'GATE=' not in line, line
        result['pairs'][case] = {k:dict(ratio=float(r),se=float(se),delta_us=float(d))
            for k,r,se,d in re.findall(r'(\w+):([\d.]+)\+/-([\d.]+):([+\-\d.]+)',line)}
    elif line.startswith('INC '):
        _, case, a, f, wall = line.split()
        result['increments'][case] = dict(A=list(map(float,a[2:].split(','))),
            FM=list(map(float,f[3:].split(','))),wall_us=list(map(float,wall[5:].split('/'))))
    elif line.startswith('EPI '):
        _, case, values = line.split()
        result['epilogue_us'][case] = {k:float(v) for k,v in (s.split('=') for s in values.split(','))}
    elif line.startswith('END shapes='):
        result['completion'] = line
    elif line.startswith('PG_TRACE_XZ85 '):
        result['cupti'] = json.loads(lzma.decompress(base64.b85decode(line.split(' ',1)[1])))
assert len(result['pairs']) == len(result['increments']) == 8, result.keys()
path = HERE / f'h200_{job}.json'
path.write_text(json.dumps(result,indent=2)+'\n')
print(path, result['payload_sha256'], result['loads'])
