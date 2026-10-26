"""Rehearse the exact weights=None import entry with the real cold-read callback."""
import hashlib
import json
from pathlib import Path
import runpy
import time
import torch
H=Path(__file__).resolve().parent
flush=torch.empty(1<<30,device='cuda',dtype=torch.uint8)
def cold():flush.view(torch.int64).sum()
s=runpy.run_path(str(H/'tokpair_payload.py'),init_globals={'lab_cold_l2':cold})
start=time.monotonic();lines=s['lab'](start+150,None,None);torch.cuda.synchronize()
assert sum('TASK=PASS' in l for l in lines)==8 and any(l.startswith('END seeds=') for l in lines)
out=dict(payload_sha256=hashlib.sha256((H/'tokpair_payload.py').read_bytes()).hexdigest(),elapsed_s=time.monotonic()-start,deadline_s=150,weights=None,lines=lines,device='GPU0 sm120',limitation='Local cold-read timing only; compiled stand-ins and JIT cache are local, so this is a deadline/correctness rehearsal rather than Hopper performance evidence.')
(H/'entry.json').write_text(json.dumps(out,indent=2)+'\n');print('\n'.join(lines),flush=True)
