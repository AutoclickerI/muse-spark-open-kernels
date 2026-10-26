import importlib.util,time,types,json,hashlib,sys
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));import local
spec=importlib.util.spec_from_file_location('sg_payload',H/'payload.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
load=p.sg_load

def local_load():
    ns=load();m=types.SimpleNamespace(**ns);local.install(m,True);ns.update(vars(m))
    for name in ('sg_prepare','sg_router','sg_run'):
        fn=ns[name];fn.__globals__.update(ns)
    return ns
p.sg_load=local_load;flush=torch.empty(1<<28,device='cuda',dtype=torch.uint8);p.lab_cold_l2=lambda:flush.view(torch.int64).sum()
a=time.monotonic();lines=p.lab(a+1800,None);print('\n'.join(lines));assert lines[-1].startswith('END complete=30/30'),lines[-3:]
assert sum(len(x)+1 for x in lines)<=9000,sum(len(x)+1 for x in lines)
(H/'lab_validation.json').write_text(json.dumps(dict(passed=True,payload_sha256=hashlib.sha256((H/'payload.py').read_bytes()).hexdigest(),elapsed=time.monotonic()-a,lines=lines),indent=2)+'\n')
