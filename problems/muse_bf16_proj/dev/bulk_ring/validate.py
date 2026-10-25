"""Native ring math, fresh graph inputs, partial ownership and local stand-ins."""
import importlib.util
import json
from pathlib import Path
import time
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
spec=importlib.util.spec_from_file_location('br_validate',ROOT/'prof/lab/dense_bulk_ring.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
gen=torch.Generator(device='cuda').manual_seed(725109);start=time.monotonic();checks=graphs=0;worst=0.;standins={}
original=dict(m.BR_CONFIGS)
for c,(r,kc,st,ks,cta) in original.items():
    if r*kc*st*2+8*st>torch.cuda.get_device_properties(0).shared_memory_per_block_optin:
        m.BR_CONFIGS[c]=(r,kc,st//2,ks,cta);standins[c]=dict(requested=list(original[c]),executed=list(m.BR_CONFIGS[c]))
# Force the full Hopper-sized persistent grid, including ranges that straddle
# a row/split boundary. All emitted partial slots must be overwritten.
for t in (1,5,8,16,17,32):
    w=torch.randn((1024,4096),device='cuda',dtype=torch.bfloat16,generator=gen)
    x=torch.randn((t,4096),device='cuda',dtype=torch.bfloat16,generator=gen)
    for config in m.BR_CONFIGS:
        for pdl in (False,True):
            op=m.BulkRingProjection(w,t,config,pdl=pdl,trace=True,grid=132*m.BR_CONFIGS[config][-1])
            # Lossless inverse packing check uses bits, not a numeric tolerance.
            raw=m.br_swizzle(op.w).permute(1,3,0,2,4).reshape_as(w)
            assert torch.equal(raw.view(torch.int16),w.view(torch.int16))
            graph,out=m.gf_graph(lambda:op(x));graphs+=1
            for seed in range(3):
                x.normal_(generator=gen)
                if seed==1:x.zero_();x[:,17]=8
                op.out.fill_(float('nan'));graph.replay()
                expected=x.double()@w.double().T
                worst=max(worst,m.gf_check(m.gf_value(out),expected,.0001,'native ring FP64 oracle'))
                assert bool(torch.isfinite(op.out).all()),'uninitialized partial slot'
                assert int(op.stamps[:,3].max()-op.stamps[:,3].min())<=1
                checks+=1
            graph.reset()
    print('BR_NATIVE',t,checks,flush=True)
record=dict(payload_sha256=m._brhash.sha256((ROOT/'prof/lab/dense_bulk_ring.py').read_bytes()).hexdigest(),
    kernel_sha256=m._brhash.sha256((HERE/'kernel.cu').read_bytes()).hexdigest(),checks=checks,graphs=graphs,
    max_relative_fp64=worst,forced_hopper_grid=[132,264],large_smem_standins=standins,
    native_stages_tested=[2,4,8],elapsed_s=time.monotonic()-start)
(HERE/'native_validation.json').write_text(json.dumps(record,indent=2)+'\n');print('BR_NATIVE_ADMISSION',json.dumps(record),flush=True)
