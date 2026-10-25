"""Run the exact packed boundary rebuild orchestration locally on GPU0."""
import base64,json,lzma,struct,time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
ns=dict(__name__='boundary_rebuild_local')
exec(compile((HERE/'rebuild_generated.py.inc').read_text(),str(HERE/'rebuild_generated.py.inc'),'exec'),ns)
flush=torch.empty(1<<30,device='cuda',dtype=torch.uint8);ns['lab_cold_l2']=lambda:flush.view(torch.int64).sum()
old=ns['br_pack']
def retain(x):
 (HERE/'local_raw.json.xz').write_bytes(lzma.compress(json.dumps(x).encode(),preset=9))
 blob=old(x);print('archive chars',len(blob),flush=True);return blob
ns['br_pack']=retain
lines=ns['lab'](time.monotonic()+800,None,None)
(HERE/'local_receipt.txt').write_text('\n'.join(lines)+'\n')
print('PASS packed rebuild lab',sum(len(s)+1 for s in lines),'chars',flush=True)
for s in lines:
 if not s.startswith('BR_REBUILD_XZ85'):print(s)
