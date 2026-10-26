"""Bind the installed FA3 exports and validate its actual setter without launching Hopper."""
import hashlib,json
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
import sgl_kernel.flash_ops
H=Path(__file__).resolve().parent;o=H/'chain';cpp=(o/'fa.cpp').read_text()
m=load_inline(name='p5b_fa_bridge_'+hashlib.sha256(cpp.encode()).hexdigest()[:14],cpp_sources=cpp,functions=['fa_bind','fa_probe','fa_main'],extra_ldflags=['-ldl'],with_cuda=True)
m.fa_bind(sgl_kernel.flash_ops.__file__)
B,L,ns=1,8,64;dev='cuda';q=torch.zeros(B*L,16,64,device=dev).to(torch.float8_e4m3fn);k=torch.zeros(300,1,2,64,device=dev).to(torch.float8_e4m3fn);v=k.clone();out=torch.empty_like(q,dtype=torch.bfloat16);lse=torch.empty(16,B*L,device=dev);oa=torch.empty(ns,16,B*L,64,device=dev);la=torch.empty(ns,16,B*L,device=dev)
seq=torch.full((B,),257+L,device=dev,dtype=torch.int32);cu=torch.arange(0,B*L+1,L,device=dev,dtype=torch.int32);pt=torch.arange(300,device=dev,dtype=torch.int32).view(B,-1);meta=torch.tensor([64,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0],device=dev,dtype=torch.int32);s=torch.ones((),device=dev)
r=m.fa_probe(q,k,v,out,lse,oa,la,seq,cu,pt,meta,s,s,L,0,ns,.684125);print(r,flush=True);assert r[0]==728 and r[1]==120 and r[3:]==[64,1,0,1]
(o/'abi_local.json').write_text(json.dumps(dict(bridge_sha256=hashlib.sha256(cpp.encode()).hexdigest(),result=r,main_launched=False,canary_pass=True),indent=2)+'\n')
