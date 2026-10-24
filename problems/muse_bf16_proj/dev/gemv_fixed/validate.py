"""Native arithmetic, graph freshness, trace ABI, real role geometry and router."""
import importlib.util
import json
from pathlib import Path
import time
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
spec=importlib.util.spec_from_file_location('gf_test',ROOT/'prof/lab/dense_gemv_fixed.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
start=time.monotonic();checks=graphs=0;worst=0.
gen=torch.Generator(device='cuda').manual_seed(72119)
for t in (1,5,8,16,17,32):
    w=torch.randn((128,8192),device='cuda',dtype=torch.bfloat16,generator=gen)
    x=torch.randn((t,8192),device='cuda',dtype=torch.bfloat16,generator=gen)
    for cfg in m.GF_CONFIGS:
        for fp32,pdl,trace in ((True,False,False),(True,True,True),(False,True,False)):
            op=m.FixedProjection(w,t,cfg,fp32,pdl,trace)
            graph,out=m.gf_graph(lambda:op(x));graphs+=1
            for seed in range(3):
                x.normal_(generator=gen)
                if seed==1:x[::8]=0;x[::8,17]=8
                graph.replay();expected=x.float()@w.float().T
                worst=max(worst,m.gf_check(m.gf_value(out),expected,.0001 if fp32 else .01,'native graph'))
                checks+=1
            if trace:m.gd_summary(op.stamps)
            graph.reset()
    print('GF_NATIVE',t,checks,flush=True)
state_m,state=m.gf_base(torch.device('cuda'))
for t in (1,8,16,32):
    for role in ('qkv_gate','o','pool_router','unpooler_slice' if t<=8 else 'unpooler'):
        w=state[role];x=torch.randn((t,w.shape[1]),device='cuda',dtype=torch.bfloat16,generator=gen)
        for cfg in m.GF_CONFIGS:
            op=m.FixedProjection(w,t,cfg,True);g,out=m.gf_graph(lambda:op(x));graphs+=1
            for seed in range(2):
                x.normal_(generator=gen);g.replay();raw=m.gf_value(out)
                worst=max(worst,m.gf_check(raw,x.float()@w.float().T,.0001,'real stage'))
                if role=='pool_router':m.gf_check(raw[:,4096:4352]+raw[:,4352:],x.float()@state['router'].T,.0001,'router')
                checks+=1
            g.reset()
    print('GF_ROLES',t,checks,flush=True)
spec=importlib.util.spec_from_file_location('gf_cell',HERE/'projection.py')
cell_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(cell_module)
for t,role in ((1,'o'),(8,'pool_router'),(16,'qkv_gate'),(32,'unpooler')):
    w=state[role];x=torch.randn((t,w.shape[1]),device='cuda',dtype=torch.bfloat16,generator=gen)
    fp32=role=='pool_router' or m.GF_PROJ[t,role]=='custom'
    cell=cell_module.Best7ProjectionCell(w,t,1,fp32=fp32)
    state['best7_stream_ops'][t,role,1,1]=cell
    plan=m.gf_plan(state_m,t);plan['projection'][role]=('stream',1,1)
    g,out=m.gf_graph(lambda:state_m._best7_project(x,role,state,plan));graphs+=1
    x.normal_(generator=gen);g.replay()
    m.gf_check(m.gf_value(out),x.float()@w.float().T,.0001 if fp32 else .01,'best7 drop-in ABI')
    checks+=1;g.reset()
report=dict(checks=checks,graphs=graphs,max_fp32_or_bf16_relative=worst,elapsed_s=time.monotonic()-start,
            payload_sha256=m._gfhash.sha256((ROOT/'prof/lab/dense_gemv_fixed.py').read_bytes()).hexdigest(),
            kernel_sha256=m._gfhash.sha256((HERE/'kernel.cu').read_bytes()).hexdigest())
(HERE/'native_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
