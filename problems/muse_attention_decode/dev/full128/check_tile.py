"""Validate the exact N128 QK/softmax/RS-PV helpers with FP8 data on the portable GPU."""
import hashlib,json
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
import build
HERE=Path(__file__).resolve().parent
source=build.source()+'\n'+(HERE/'tile.cu.inc').read_text();tag=hashlib.sha256(source.encode()).hexdigest()[:12]
major,minor=torch.cuda.get_device_capability();arch='90a' if major==9 else f'{major}{minor}'
m=load_inline(name=f'full128_math_{tag}_{arch}',cpp_sources='#include <cstdint>\nvoid full128_tile_forward(int64_t,int64_t,int64_t,int64_t,double,int64_t,int64_t);',cuda_sources=source,functions=['full128_tile_forward'],extra_cuda_cflags=['-O3',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
rows=[]
for seed in range(4):
    torch.manual_seed(1200+seed)
    q=torch.randn((2,64,64),device='cuda').to(torch.float8_e4m3fn)
    k=torch.randn((2,128,64),device='cuda').to(torch.float8_e4m3fn)
    v=torch.randn_like(k,dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    for masked in (0,1):
        for sc in (.125,.7):
            out=torch.empty((2,64,64),device='cuda')
            def fn():m.full128_tile_forward(q.data_ptr(),k.data_ptr(),v.data_ptr(),out.data_ptr(),sc*1.4426950408889634,masked,torch.cuda.current_stream().cuda_stream)
            fn();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):fn()
            q.copy_(torch.randn_like(q,dtype=torch.bfloat16).to(q.dtype));graph.replay();torch.cuda.synchronize()
            logits=q.float()@k.float().transpose(-1,-2)*sc
            if masked:logits.masked_fill_(torch.arange(128,device='cuda')[None,None,:]>=96+torch.arange(64,device='cuda')[None,:,None]//8,-torch.inf)
            want=logits.softmax(-1)@v.float();err=float((out-want).norm()/want.norm());assert err<.045,(seed,masked,sc,err)
            rows.append(dict(seed=seed,masked=masked,scale=sc,error=err));print('PASS',rows[-1],flush=True)
(HERE/'tile_local.json').write_text(json.dumps(rows,indent=2)+'\n')
