"""Rehearse the unchanged round3/4 pair with the wrapper's shared150s deadline."""
import hashlib
import json
from pathlib import Path
import runpy
import time
import torch
H=Path(__file__).resolve().parent;B=H.parent
flush=torch.empty(1<<30,device='cuda',dtype=torch.uint8)
def cold():flush.view(torch.int64).sum()
started=time.monotonic();end=started+150;out={}
for i,(name,file) in enumerate((('round3','tail_payload.py'),('round4','ws_payload.py'))):
 tick=time.monotonic();deadline=tick+max(0,end-tick)/(2-i)
 source=B/name/file;s=runpy.run_path(str(source),init_globals={'lab_cold_l2':cold})
 lines=s['lab'](deadline,None,None);torch.cuda.synchronize()
 expected=10 if name=='round3' else 8;assert sum('TASK=PASS' in line for line in lines)==expected
 out[name]=dict(payload_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),allocated_s=deadline-tick,elapsed_s=time.monotonic()-tick,lines=lines)
 print(name,'allocated',deadline-tick,'used',out[name]['elapsed_s'],lines[-1],flush=True)
out['total_s']=time.monotonic()-started;out['limitation']='GPU0 sm120 with local JIT cache. Native Hopper compilation is still charged to the same150s budget; actual seed counts may be lower.'
(H/'packed_entry.json').write_text(json.dumps(out,indent=2)+'\n');print('COMPLETE',out['total_s'],flush=True)
