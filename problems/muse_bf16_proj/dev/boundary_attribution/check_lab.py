"""Validate batch orchestration on sm120 portable paths; no H200 performance claim."""
import base64,json,lzma,struct,time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
ns=dict(__name__='boundary_local_lab')
exec(compile((HERE/'boundary_generated.py.inc').read_text(),str(HERE/'boundary_generated.py.inc'),'exec'),ns)
ns['pa_changed_modules']=lambda module:{}
flush=torch.empty(1<<30,device='cuda',dtype=torch.uint8)
ns['lab_cold_l2']=lambda:flush.view(torch.int64).sum()
original_pack=ns['bd_pack']
def retain(confirmation,boundaries):
 (HERE/'local_raw.json.xz').write_bytes(lzma.compress(json.dumps(dict(confirmation=confirmation,boundaries=boundaries)).encode(),preset=9))
 blob=original_pack(confirmation,boundaries)
 (HERE/'local_archive.txt').write_text('BD_BATCH_XZ85 '+blob+'\n')
 print('archive size',len(blob),flush=True)
 return blob
ns['bd_pack']=retain
lines=ns['lab'](time.monotonic()+800,None,None)
blob=next(s.split(' ',1)[1] for s in lines if s.startswith('BD_BATCH_XZ85 '))
from report import decode
data=decode(blob)
assert len(data['confirmation']['cells'])==12 and len(data['boundaries'])==4
assert all(not c['gates'] and c['sample_rows']==16 for c in data['confirmation']['cells']+data['boundaries'])
(HERE/'local_batch.json').write_text(json.dumps(data,indent=2)+'\n')
print('PASS 12 confirmation +4 boundary fixtures,16 seeds each; archive bytes',sum(len(s)+1 for s in lines),flush=True)
for s in lines:
 if not s.startswith('BD_BATCH_XZ85'):print(s)
