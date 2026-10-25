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
src=(ROOT/'prof/lab/full_poly_timeline.py').read_text();name='<cs_sliding_smoke>'
linecache.cache[name]=(len(src),None,src.splitlines(True),name);ns={}
exec(compile(src,name,'exec'),ns)
st=ns['tl_prepare_selected'](host,weights)
ns['TL_EXISTING_HOST']=(host,st)
buf=torch.empty(1<<28,device='cuda',dtype=torch.uint8);ns['lab_cold_l2']=lambda:buf.view(torch.int64).sum()
ns['PF_SEEDS']=ns['PF_SEEDS'][:2];ns['PF_ROUNDS']=4;ns['PF_CLOCKS']=False
if '--quick' in sys.argv:ns['PF_CASES']=(16,)
try:
    for line in ns['lab'](time.monotonic()+220,weights,st):print(line,flush=True)
finally:(HERE/'timeline_local.json').write_text(json.dumps(ns['PF_RESULTS'],indent=2)+'\n')
assert len(ns['PF_RESULTS'])==len(ns['PF_CASES'])
