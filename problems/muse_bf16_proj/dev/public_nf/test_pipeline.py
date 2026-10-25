"""Correctness, graph/PDL, neighboring M, arbitrary split sizes, exact escapes."""
import json
from pathlib import Path
import torch
from pipeline_x import cd_encode,cd_decode,cd_module,cd_gemv
H=Path(__file__).resolve().parent
cd_module(tuple((nt,1,2,3) for nt in (1,2,4)))
records=[]
for seed in (61791,14357,92003):
    g=torch.Generator(device='cuda').manual_seed(seed)
    for N,K in ((256,1024),(1024,4096),(2304,8192),(4608,8192)):
        w=(torch.randn((N,K),device='cuda',generator=g)/K**.5).to(torch.bfloat16)
        w[0,0]=0.;w[3,139]=-0.;w[19,27]=32.
        c=cd_encode(w,kmajor=True)
        assert torch.equal(cd_decode(c).view(torch.int16),w.view(torch.int16))
        for M in (1,5,8,9,16,17,32):
            x=torch.randn((M,K),device='cuda',generator=g).to(torch.bfloat16)
            x[-1]=0;x[-1,27]=16
            ref=x.double()@w.double().T
            for S in sorted({8,11,min(22,K//64)}):
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):p,_=cd_gemv(x,c,S,NW=4,V=3,pdl=True)
                graph.replay();torch.cuda.synchronize()
                err=float((p.double().sum(0)-ref).norm()/ref.norm())
                assert err<3e-5,(seed,N,K,M,S,err)
                records.append(dict(seed=seed,N=N,K=K,M=M,S=S,error=err))
        print('PASS',seed,N,K,flush=True)
(H/'pipeline_validation.json').write_text(json.dumps(dict(passed=True,cases=len(records),max_error=max(r['error'] for r in records),records=records),indent=2)+'\n')
