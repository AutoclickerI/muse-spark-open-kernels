"""Exercise the whole-layer lab's A/A2/PP graph capture and namespace binding on local inputs."""
import importlib.util
import linecache
from pathlib import Path
import sys
import time
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_decode_layer')]
import local_eval
m=local_eval.load_module('submission',HERE/'out/s11_partial_port.py')
ref=local_eval.load_module('reference',ROOT/'problems/muse_decode_layer/reference.py')
ref._fa3_scheduler_metadata=lambda *a,**k:None
weights=ref.generate_weights(ref.WEIGHTS_SEED)
m._ap_fa3_scheduler_metadata=lambda *a,**k:None
state=m.prepare(weights)
src=(ROOT/'prof/lab/attention_p5_layer.py').read_text()
fname='<p5_layer_smoke>'
linecache.cache[fname]=(len(src),None,src.splitlines(True),fname)
ns=dict(m.__dict__)
buf=torch.empty(1<<28,dtype=torch.uint8,device='cuda')
ns['lab_cold_l2']=lambda:buf.view(torch.int64).sum()
exec(compile(src,fname,'exec'),ns)
ns['AB_SEEDS']=ns['AB_SEEDS'][:2]
ns['AB_CASES']=((1,1,2048),(1,1,0))
ns['AB_ROUNDS']=4
for line in ns['lab'](time.monotonic()+90,weights,state):print(line,flush=True)
