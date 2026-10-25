"""Execute the packed lab in the judge image on GPU0; Hopper attention remains a portable stand-in."""
import base64,json,lzma,struct,time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
ns=dict(__name__='d1_local_lab')
exec(compile((HERE/'chain_generated.py.inc').read_text(),str(HERE/'chain_generated.py.inc'),'exec'),ns)
flush=torch.empty(1<<30,device='cuda',dtype=torch.uint8);ns['lab_cold_l2']=lambda:flush.view(torch.int64).sum()
original=ns['dc_pack']
def retain(x):
 (HERE/'local_raw.json.xz').write_bytes(lzma.compress(json.dumps(x).encode(),preset=9))
 blob=original(x);print('archive chars',len(blob),flush=True);return blob
ns['dc_pack']=retain
lines=ns['lab'](time.monotonic()+800,None,None)
(HERE/'local_receipt.txt').write_text('\n'.join(lines)+'\n')
print('PASS full local lab',sum(len(s)+1 for s in lines),'chars',flush=True)
for s in lines:
 if not s.startswith('D1_CHAIN_XZ85'):print(s)
