"""Exact payload screen with forced layer dispatch; native cluster gangs use explicit C1 stand-ins."""
import hashlib,importlib.util,json,linecache,sys,time,types
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_decode_layer')]
import torch
payload_path=Path(sys.argv[1]) if len(sys.argv)>1 else H/'out/sliding_round2_lab.py'
s=payload_path.read_text();fn='<p5r2-boot-local>';linecache.cache[fn]=(len(s),None,s.splitlines(True),fn)
ns={};exec(compile(s,fn,'exec'),ns)
# h200_lab cold-READ helper, same cache protocol as the queued packet.
buf=torch.empty(1<<30,device='cuda',dtype=torch.uint8)
ns['lab_cold_l2']=lambda:buf.view(torch.int64).sum()
spec=importlib.util.spec_from_file_location('p5r2_force',R/'problems/muse_decode_layer/dev/best7/force.py');force=importlib.util.module_from_spec(spec);spec.loader.exec_module(force)
orig_host=ns['host_module'];orig_attention=ns['attention'];orig_module=ns['module'];standins=[]
def host_module():
    host=orig_host();force.install(host,False,False);host._ap_fa3_scheduler_metadata=lambda *a,**kw:None
    return host

def attention(data,qkv=None,kind='last',qt=1,ns=8):
    if kind in ('dsm','dsm_old'):
        standins.append((data['config']['batch'],data['config']['q_len'],kind,qt,ns));ns=1
    return orig_attention(data,qkv,kind,qt,ns)

class Proxy:
    def __init__(self,real):self.real=real
    def sliding_resources(self,L,qt,nk,cs):return self.real.sliding_resources(L,qt,nk,1)
    def sliding_forward(self,*args):return self.real.sliding_forward(*args)

def module(kind):return Proxy(orig_module(kind)) if kind in ('dsm','dsm_old') else orig_module(kind)
ns['host_module']=host_module;ns['attention']=attention;ns['module']=module
ns['SEEDS']=ns['SEEDS'][:2];ns['ROUNDS']=2
lines=ns['lab'](time.monotonic()+1500,None,None)
assert len(ns['RESULTS'])==len(ns['CASES'])
assert all(all(v==2 for v in r['task_passed'].values()) for r in ns['RESULTS'])
archive=next(x for x in lines if x.startswith('R2_XZ85 '));print('ARCHIVE chars',len(archive))
print('\n'.join(x for x in lines if not x.startswith('R2_XZ85 ')))
(H/'out'/(payload_path.stem+'_admission.json')).write_text(json.dumps(dict(payload_sha256=hashlib.sha256(s.encode()).hexdigest(),results=ns['RESULTS'],standins=standins,local_only=True),indent=2)+'\n')
print('PASS exact payload: selected shapes, all arms, full reference 20pct margin,2seeds x2cold rounds, forced Hopper layer glue, DSM C1 stand-in')
