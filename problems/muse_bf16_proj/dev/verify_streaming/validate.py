"""Exercise TMA, split-K publication and fresh-input graphs on the local GPU."""
import importlib.util
import json
import time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('vs_projection', HERE/'projection.py')
vs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vs)


def check(op,x,w,probe=False):
    want = x.float()@w.float().T
    out = op(x)
    torch.cuda.synchronize()
    rel = float((out.float()-want).norm()/want.norm().clamp_min(1e-12))
    row = float(((out.float()-want).norm(dim=1)/want.norm(dim=1).clamp_min(1e-10)).max())
    limit = 1e-4 if op.fp32 else (.001 if probe else .01)
    assert bool(torch.isfinite(out).all()) and rel<=limit and row<=limit*5,(op.code,op.splits,rel,row)
    assert not bool(op.counters.any())
    return rel,row


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    start = time.monotonic()
    checked = graphs = descriptors = 0
    worst = [0.,0.]
    for fp32 in (False,True):
        w = torch.randn((257,1024),device='cuda',dtype=torch.bfloat16)
        x = torch.randn((129,1024),device='cuda',dtype=torch.bfloat16)
        for code,(_,_,bk,_,_) in vs.VS_CONFIGS.items():
            for splits in (1,2,4):
                if w.shape[1]%(bk*splits):
                    continue
                op = vs.VerifyStreamingProjection(w,len(x),code,splits,fp32)
                check(op,x,w)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    out = op(x)
                for j in range(3):
                    x.normal_()
                    if j == 1:
                        x[::3]*=8
                        x[:,::127]*=16
                    if j == 2:
                        x.zero_()
                        rows = torch.arange(len(x),device='cuda')
                        x[rows,(rows*1031+17)%1024] = torch.exp2((rows%7-3).float()).to(x.dtype)
                    g.replay()
                    want = x.float()@w.float().T
                    rel = float((out.float()-want).norm()/want.norm().clamp_min(1e-12))
                    row = float(((out.float()-want).norm(dim=1)/want.norm(dim=1).clamp_min(1e-10)).max())
                    limit = 1e-4 if fp32 else (.001 if j==2 else .01)
                    assert bool(torch.isfinite(out).all()) and rel<limit and row<limit*5,(code,splits,j,rel,row)
                    assert not bool(op.counters.any())
                    worst = [max(worst[0],rel),max(worst[1],row)]
                    checked += 1
                g.reset()
                graphs += 1
                print('PASS',fp32,code,splits,flush=True)
    # Exact decode banks and all three verify loads, including truncated unpooler at M8.
    for m in (8,64,128,256):
        for n,k,fp32 in ((2304,8192,False),(8192,1024,False),(4608,8192,True),
                         (1024 if m<=8 else 8192,4096,False)):
            w = torch.randn((n,k),device='cuda',dtype=torch.bfloat16)/(k**.5)
            x = torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
            for code in (2,4,6,7,8,14):
                choices = [s for c,s in vs.vs_choices(m,n,k) if c==code]
                for splits in choices:
                    op = vs.VerifyStreamingProjection(w,m,code,splits,fp32)
                    for _ in range(2):
                        x.normal_()
                        rr,rw = check(op,x,w)
                        worst = [max(worst[0],rr),max(worst[1],rw)]
                        checked += 1
            print('SHAPE_PASS',m,n,k,fp32,flush=True)
    # The Hopper ABI can encode descriptors locally without executing WGMMA.
    native = vs.vs_library('9.0a')
    for m in (8,128,256):
        for n,k in ((2304,8192),(8192,1024),(4608,8192),(8192,4096)):
            w = torch.empty((n,k),device='cuda',dtype=torch.bfloat16)
            x = torch.empty((m,k),device='cuda',dtype=torch.bfloat16)
            for code,splits in vs.vs_choices(m,n,k):
                op = vs.VerifyStreamingProjection(w,m,code,splits,n==4608,lib=native)
                op.launch(x,descriptor_only=True)
                descriptors += 1
    # Explicit dispatch ABI, incorrect precision/stride rejection.
    w = torch.randn(257,1024,device='cuda',dtype=torch.bfloat16)
    x = torch.randn(8,1024,device='cuda',dtype=torch.bfloat16)
    state = {'qkv_gate':w}
    vs.vs_prepare_dispatch(state,{(8,'qkv_gate'):(2,2)})
    p,s = vs.vs_project(x,w,'qkv_gate',state)
    assert p.shape==(1,8,257) and s==1 and p.dtype==torch.bfloat16
    rejects = 0
    for bad in (x.float(),x.T.contiguous().T,x[:7]):
        try:
            vs.vs_project(bad,w,'qkv_gate',state)
        except ValueError:
            rejects += 1
    assert rejects==3
    result = dict(device=torch.cuda.get_device_name(),fresh_checks=checked,graphs=graphs,
                  native_descriptors=descriptors,abi_rejections=rejects,maxrel=worst[0],maxrow=worst[1],
                  elapsed_s=time.monotonic()-start,native_execution=False)
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('VALIDATION',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
