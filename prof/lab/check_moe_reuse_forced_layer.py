"""Force portable reuse producers inside the final layer's four scored verify graph cases."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'tools'),str(root/'problems/muse_decode_layer')]
import local_eval as ev
p=root/'problems/muse_decode_layer/cand_r3_v9_reuse.py'
ref=ev.load_module('reference',root/'problems/muse_decode_layer/reference.py')
if hasattr(ref,'_fa3_scheduler_metadata'):ref._fa3_scheduler_metadata=lambda *a,**kw:None
w=ref.generate_weights(ref.WEIGHTS_SEED)
mod=ev.load_module('forced_reuse_layer',p)
state=mod.prepare(ev.clone(w))
state['moe_bank']['moe_reuse_plan']={128:(4,0,2),256:(16,1,2)}
print('FORCED',state['moe_bank']['moe_reuse_plan'],flush=True)
side=SimpleNamespace(call=lambda data:mod.kernel_fn(data,state),name='forced')
problem=ev.Problem(ref,w);timer=ev.Timer()
spec=json.loads((root/'upstream/api/problems/muse_decode_layer.json').read_text())
rows=[]
for case in spec['benchmarks']:
    if case['load'] not in ('verify 16','verify 32'):continue
    row=ev.time_shape(problem,[side],case,20,3,timer)
    assert row['ok'],row
    rows.append(row);print('PASS',case['load'],case.get('layer'),flush=True)
assert len(rows)==4
out=dict(source=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),choices=state['moe_bank']['moe_reuse_plan'],shapes=rows)
(root/'runs/screen/v9_reuse/final_forced.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS 4 forced verify layer cases, 20 replays and 3 fresh inputs each; existing static/13-test screen applies to the same source bytes')
