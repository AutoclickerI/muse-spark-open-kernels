"""Exact import-time lab rehearsal and fixed-width seed archive round trip."""
import base64
import hashlib
import json
from pathlib import Path
import runpy
import struct
import time
import torch
H=Path(__file__).resolve().parent
flush=torch.empty(1<<30,device='cuda',dtype=torch.uint8)
def cold():flush.view(torch.int64).sum()
n=runpy.run_path(str(H/'s15_review_payload.py'),init_globals={'lab_cold_l2':cold})
t=time.monotonic();lines=n['lab'](t+150,None,None);torch.cuda.synchronize()
assert sum('TASK=PASS' in l for l in lines)==12
archive=json.loads(next(l[6:] for l in lines if l.startswith('SEEDS ')))
raw=base64.b85decode(archive['data']);count=2*archive['seeds']*(len(archive['names'])+2)
values=struct.unpack('<'+'d'*count,raw);assert len(raw)==count*8
assert len('\n'.join(lines))<9000
out=dict(payload_sha256=hashlib.sha256((H/'s15_review_payload.py').read_bytes()).hexdigest(),elapsed_s=time.monotonic()-t,deadline_s=150,weights=None,lines=lines,archive_bytes=len(raw),receipt_characters=len('\n'.join(lines)),local_standin='sm120 real math/fence/acquire glue; S4 usesS3, readiness consumer grid half SMs; no native H200 performance inference.')
(H/'entry.json').write_text(json.dumps(out,indent=2)+'\n');print('\n'.join(lines),flush=True)
