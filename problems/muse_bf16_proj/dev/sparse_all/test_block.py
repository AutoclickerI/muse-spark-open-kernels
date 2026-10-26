"""Independent full-BF16 proof for the grouped Triton reader, all real matrix roles."""
from pathlib import Path
import importlib.util,linecache,json,hashlib,math
import torch
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('dc_block_test',H/'candidate.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
source=(H/'block_reader.py.inc').read_text();file='<dc_block_test>'
linecache.cache[file]=(len(source),None,source.splitlines(True),file);exec(compile(source,file,'exec'),m.__dict__)
g=torch.Generator(device='cuda').manual_seed(61791);rows=[]
def dense(n,k):return (torch.randn((n,k),device='cuda',generator=g)/math.sqrt(k)).bfloat16()
w=dict(qkv_gate=dense(2304,8192),o=dense(8192,1024),pooler=dense(4096,8192),router=.045*torch.randn((256,8192),device='cuda',generator=g),unpooler=dense(8192,4096))
for name,a in m.dc_role_weights(w).items():
    a=a.clone();a[3,139]=0.;a[11,211]=-0.;a[19,27]=3.;a[23,39]=torch.finfo(torch.bfloat16).tiny
    bank,_=m.dc_encode_cd(m.cd_encode(a));n,k=a.shape;out=torch.empty_like(a)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):m.dc_block_restore[(n//64,k//128)](bank['coded'],out,N=n,K=k,BN=64,BK=128,num_warps=4)
    graph.replay();torch.cuda.synchronize();assert torch.equal(out.view(torch.int16),a.view(torch.int16)),name
    rows.append(dict(role=name,n=n,k=k,exact=True));print('PASS',name,flush=True)
(H/'block_validation.json').write_text(json.dumps(dict(passed=True,source_sha256=hashlib.sha256(source.encode()).hexdigest(),rows=rows),indent=2)+'\n')
