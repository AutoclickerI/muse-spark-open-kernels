"""Exercise the paired-lab orchestration locally; Hopper machine code is audited separately."""
import json
from pathlib import Path
import sys
import time
import torch

HERE=Path(__file__).resolve().parent
followup='--followup' in sys.argv
shared='--shared-jit' in sys.argv
src=(HERE/('shared_followup_generated.py.inc' if shared else 'followup_generated.py.inc' if followup else 'lab_body_generated.py.inc')).read_text()
ns=dict(__name__='pdl_local_lab')
exec(compile(src,str(HERE/'lab_body_generated.py.inc'),'exec'),ns)
# sm120 uses S14's portable plan. Do not compile/launch Hopper-only vq8 here.
ns['pa_changed_modules']=lambda module:{}
flush=torch.empty(1<<30,device='cuda',dtype=torch.uint8)
ns['lab_cold_l2']=lambda:flush.view(torch.int64).sum()
lines=ns['lab'](time.monotonic()+650,None,None)
for line in lines:print(line,flush=True)
archive=next(line.split(' ',1)[1] for line in lines if line.startswith('PDL_AUDIT_XZ85 '))
import base64,lzma
data=json.loads(lzma.decompress(base64.b85decode(archive)))
(HERE/('local_shared_followup.json' if shared else 'local_followup.json' if followup else 'local_lab.json')).write_text(json.dumps(data,indent=2)+'\n')
assert len(data['cells'])==12
assert all(c['seeds']==16 and not c['gates'] for c in data['cells'])
assert sum(len(s)+1 for s in lines)<9000,'archive exceeds the single-payload wrapper cap'
print('PASS all 12 paired lab fixtures,16 seeds each; portable paths only')
