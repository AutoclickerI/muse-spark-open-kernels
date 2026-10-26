"""Run the direct library-replacement structure on SM120, including graph replay."""
import importlib.util,json,hashlib,math
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;block='--block' in __import__('sys').argv;candidate=H/('block_candidate.py' if block else 'candidate.py')
sp=importlib.util.spec_from_file_location('dc_direct_test',candidate);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
g=torch.Generator(device='cuda').manual_seed(61791)
def dense(n,k):return (torch.randn((n,k),device='cuda',generator=g)/math.sqrt(k)).bfloat16()
w=dict(qkv_gate=dense(2304,8192),o=dense(8192,1024),pooler=dense(4096,8192),router=.045*torch.randn((256,8192),device='cuda',generator=g),unpooler=dense(8192,4096))
roles=m.dc_role_weights(w);rows=[]
for name,a in roles.items():
    if name in ('qkv3','unpooler_slice'):continue
    c,_=m.dc_encode_cd(m.cd_encode(a))
    for t in (128,256):
        x=torch.randn((t,a.shape[1]),device='cuda',generator=g).bfloat16()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):p,s=m.dc_vdsk_project(x,c,True)
        for seed in (61791,14357,92003):
            g.manual_seed(seed);x.copy_(torch.randn(x.shape,device='cuda',generator=g).bfloat16());x[-1]=0.;x[-1,27]=16
            graph.replay();ref=x.double()@a.double().T;got=p.double().sum(0)
            err=float((got-ref).norm()/ref.norm());probe=float((got[-1]-ref[-1]).norm()/ref[-1].norm())
            assert err<3e-5 and probe<1e-5,(name,t,seed,err,probe)
            rows.append(dict(role=name,t=t,seed=seed,error=err,probe=probe));print('PASS',name,t,seed,err,flush=True)
(H/('block_direct_validation.json' if block else 'direct_validation.json')).write_text(json.dumps(dict(passed=True,sha256=hashlib.sha256((candidate).read_bytes()).hexdigest(),rows=rows),indent=2)+'\n')
