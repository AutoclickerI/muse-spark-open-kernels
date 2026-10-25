"""Exercise the exact embedded layer/timeline payload on the portable GPU before H200."""
import sys,time,json,linecache
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_decode_layer')]
import local_eval
host=local_eval.load_module('submission',ROOT/'problems/muse_decode_layer/cand_r3_best7_s11_final.py')
ref=local_eval.load_module('reference',ROOT/'problems/muse_decode_layer/reference.py')
ref._fa3_scheduler_metadata=lambda *a,**kw:None;host._ap_fa3_scheduler_metadata=lambda *a,**kw:None
weights=ref.generate_weights(ref.WEIGHTS_SEED)
src=(ROOT/'prof/lab/gated_sliding_timeline.py').read_text();name='<cs_sliding_smoke>'
linecache.cache[name]=(len(src),None,src.splitlines(True),name);ns={}
exec(compile(src,name,'exec'),ns)
st=ns['tl_prepare_selected'](host,weights)
ns['TL_EXISTING_HOST']=(host,st)
buf=torch.empty(1<<28,device='cuda',dtype=torch.uint8);ns['lab_cold_l2']=lambda:buf.view(torch.int64).sum()
ns['CS_SEEDS']=ns['CS_SEEDS'][:2];ns['CS_ROUNDS']=4
if '--quick' in sys.argv:ns['CS_CASES']=((1,1),)
try:
    for line in ns['lab'](time.monotonic()+220,weights,st):print(line,flush=True)
finally:(HERE/'timeline_local.json').write_text(json.dumps(ns['CS_RESULTS'],indent=2)+'\n')
assert len(ns['CS_RESULTS'])==len(ns['CS_CASES'])
