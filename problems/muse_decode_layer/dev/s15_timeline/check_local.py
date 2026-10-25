"""Run unchanged S15 plus observational capture locally with normal or forced Hopper stand-ins."""
import argparse,hashlib,importlib.util,json,lzma,sys,time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]

def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
sys.path.insert(0,str(ROOT/'problems/muse_decode_layer'))
p=argparse.ArgumentParser();p.add_argument('--forced',action='store_true');a=p.parse_args()
m=load('s15_timeline_host',ROOT/'problems/muse_decode_layer/cand_r3_best7_s15.py')
if a.forced:load('timeline_force',HERE.parent/'best7/force.py').install(m,False,False)
load('timeline_local_standins',HERE.parent/'s15/local_standins.py').install(m)
m._ap_fa3_scheduler_metadata=lambda *a,**kw:None
ns=dict(kernel_fn=m.kernel_fn,__name__='s15_timeline_local')
exec(compile((HERE/'timeline_generated.py.inc').read_text(),str(HERE/'timeline_generated.py.inc'),'exec'),ns)
ref=load('timeline_ref',ROOT/'problems/muse_decode_layer/reference.py')
weights=ref.generate_weights(5353,router_seed=197731);t0=time.monotonic();state=m.prepare(weights);prepare_s=time.monotonic()-t0
flush=torch.empty(1<<30,device='cuda',dtype=torch.uint8);ns['lab_cold_l2']=lambda:flush.view(torch.int64).sum()
old=ns['t15_pack']
def retain(x):
 (HERE/('forced_raw.json.xz' if a.forced else 'local_raw.json.xz')).write_bytes(lzma.compress(json.dumps(x).encode(),preset=9))
 blob=old(x);print('archive chars',len(blob),flush=True);return blob
ns['t15_pack']=retain
lines=ns['lab'](time.monotonic()+650,weights,state)
(HERE/('forced_receipt.txt' if a.forced else 'local_receipt.txt')).write_text('\n'.join(lines)+'\n')
print('PASS local', 'forced' if a.forced else 'normal','prepare_s',round(prepare_s,1),'receipt chars',sum(len(s)+1 for s in lines),flush=True)
for s in lines:
 if not s.startswith('S15_TIMELINE_XZ85'):print(s)
