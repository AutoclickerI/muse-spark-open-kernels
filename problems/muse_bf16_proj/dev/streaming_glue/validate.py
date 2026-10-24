"""Exercise native shared/scalar consumers and fused continuations on sm_120."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch

HERE=Path(__file__).resolve().parent
payload=HERE.parent/'streaming_glue_lab.py'
spec=importlib.util.spec_from_file_location('streaming_glue',payload)
sg=importlib.util.module_from_spec(spec);spec.loader.exec_module(sg)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
begin=time.monotonic();device='cuda';checked=graphs=0;worst=0.
gen=torch.Generator(device=device).manual_seed(8931)
def rand(*s,dtype=torch.float32):return torch.randn(s,device=device,dtype=dtype,generator=gen)
weights={name:rand(8192,dtype=torch.bfloat16)*.1 for name in
         ('attn_residual_gate','ffn_residual_gate','pre_ffn_norm','next_pre_attn_norm','post_ffn_norm')}
weights.update(pre_expert_norm=rand(4096,dtype=torch.bfloat16)*.1,router_bias=rand(256)*.055)
params=sg.sg_params(weights)
sg.sp_library()
# Boundary tails and every compiled consumer, including T=128 on M=1.
for m in (1,7,8,16,32,33,128,130,256):
    w=rand(257,896,dtype=torch.bfloat16)/896**.5
    x,stream,start=sg.sg_inputs(m,896,93,device)
    for phase in ('o','unpool'):
        ref,eff,raw=sg.sg_oracle(phase,x,stream,start,w,params[phase])
        for cfg in sg.SP_CONFIGS:
            if cfg%200<10 and m>8:continue
            for splits in (1,2,4):
                op=sg.FusedStreamingProjection(w,m,cfg,splits,phase,params[phase])
                out=op(x,stream,start)
                worst=max(worst,sg.sg_check(phase,out,ref,op.out,raw,eff))
                if bool(op.ready.any()) or bool(op.counters.any()):raise RuntimeError('dirty counters')
                checked+=1
                if splits==4 and cfg%200 in (1,2,10,12,14,22,110,112,114,122):
                    graph,out=sg.sg_graph(lambda:op(x,stream,start))
                    for seed in (13,70,45):
                        xx,ss,st=sg.sg_inputs(m,896,seed,device)
                        x.copy_(xx);stream.copy_(ss);start.copy_(st);graph.replay()
                        rr,ee,pp=sg.sg_oracle(phase,x,stream,start,w,params[phase])
                        sg.sg_check(phase,out,rr,op.out,pp,ee)
                    if bool(op.ready.any()) or bool(op.counters.any()):raise RuntimeError('dirty graph counters')
                    graph.reset();graphs+=1
                    xx,ss,st=sg.sg_inputs(m,896,93,device)
                    x.copy_(xx);stream.copy_(ss);start.copy_(st)
    print(f'TAIL_PASS m={m} checked={checked} graphs={graphs}',flush=True)

weights.update(o=rand(8192,1024,dtype=torch.bfloat16)/32,
               unpooler=rand(8192,4096,dtype=torch.bfloat16)/64,
               qkv_gate=rand(2304,8192,dtype=torch.bfloat16)/8192**.5,
               pooler=rand(4096,8192,dtype=torch.bfloat16)/8192**.5,
               router=rand(256,8192)*.045)
hi=weights['router'].bfloat16();lo=(weights['router']-hi.float()).bfloat16()
pool=torch.cat((weights['pooler'],hi,lo)).contiguous()
selected={}
for m in (1,7,8,16,32,33,128,130,256):
    for name in ('o','pool','pool_norm','unpool'):
        phase='pool' if name=='pool_norm' else name
        w=weights['o'] if phase=='o' else pool if phase=='pool' else weights['unpooler'][:1024].contiguous() if m<=8 else weights['unpooler']
        inputs=sg.sg_inputs(m,w.shape[1],93,device,name=='pool_norm');x,stream,start=inputs
        ref,eff,raw=sg.sg_oracle(phase,*inputs,w,params[phase],name=='pool_norm')
        router=weights['router'] if phase=='pool' else None
        for cfg,splits in sg.sg_choices(m,w.shape[1]):
            op=sg.FusedStreamingProjection(w,m,cfg,splits,phase,params[phase],name=='pool_norm')
            worst=max(worst,sg.sg_check(phase,op(*inputs),ref,op.out,raw,eff,router))
            if bool(op.ready.any()) or bool(op.counters.any()):raise RuntimeError('dirty full counters')
            checked+=1
        cfg=1 if m==1 else 112 if m<=32 else 121
        for mode in (0,200):
            op=sg.FusedStreamingProjection(w,m,cfg+mode,4,phase,params[phase],name=='pool_norm')
            old=sg.CublasGlue(w,m,phase,params[phase],name=='pool_norm')
            graph,out=sg.sg_graph(lambda:op(*inputs));oldgraph,oldout=sg.sg_graph(lambda:old(*inputs))
            for seed in (19,42,99):
                fresh=sg.sg_inputs(m,w.shape[1],seed,device,name=='pool_norm')
                for a,b in zip(inputs,fresh):a.copy_(b)
                graph.replay();oldgraph.replay()
                rr,ee,pp=sg.sg_oracle(phase,*inputs,w,params[phase],name=='pool_norm')
                sg.sg_check(phase,out,rr,op.out,pp,ee,router)
                sg.sg_check(phase,oldout,rr,None,pp,ee,router)
            if bool(op.ready.any()) or bool(op.counters.any()):raise RuntimeError('dirty full graph counters')
            graph.reset();oldgraph.reset();graphs+=2
        selected[m,name]=(0.,cfg,4,0.,0.)
        print(f'FULL_PASS m={m} phase={name} checked={checked} graphs={graphs}',flush=True)
    flush=torch.zeros(256<<20,device=device,dtype=torch.uint8)
    for mode in (0,200):
        recipes={k:(v[0],v[1]+mode,*v[2:]) for k,v in selected.items()}
        chain=sg.sg_chain(time.monotonic()+20,weights,params,pool,m,recipes,flush)
        if any('REJECT' in l or 'missing_winner' in l for l in chain):raise RuntimeError('\n'.join(chain))
        print('\n'.join(chain),flush=True)
        graphs+=3

# Complete score ties must choose the smaller expert IDs, including a local
# expert range that wraps at 256. Then test cancellation in both boundaries.
tie=pool.clone();tie[4096:]=0
tieparams=dict(params['pool'],bias=torch.zeros_like(params['pool']['bias']))
for m,cfg in ((1,201),(8,312),(33,321)):
    xx,ss,st=sg.sg_inputs(m,8192,0,device);st.fill_(253)
    op=sg.FusedStreamingProjection(tie,m,cfg,4,'pool',tieparams)
    rr,ee,pp=sg.sg_oracle('pool',xx,ss,st,tie,tieparams)
    sg.sg_check('pool',op(xx,ss,st),rr,op.out,pp,ee)
    checked+=1
for m,cfg in ((1,201),(8,312),(33,321)):
    for phase in ('o','unpool'):
        w=weights['o'] if phase=='o' else weights['unpooler'][:1024].contiguous() if m<=8 else weights['unpooler']
        xx,ss,st=sg.sg_inputs(m,w.shape[1],71,device)
        raw=torch.mm(xx,w.t(),out_dtype=torch.float32)
        br=torch.nn.functional.pad(raw,(0,8192-w.shape[0]))
        if phase=='unpool':br*=params[phase]['post']
        ss.copy_(-params[phase]['beta']*sg.sg_norm(br,1.,1.e-8)/params[phase]['alpha']+.001*ss)
        rr,ee,pp=sg.sg_oracle(phase,xx,ss,st,w,params[phase])
        op=sg.FusedStreamingProjection(w,m,cfg,4,phase,params[phase])
        out=op(xx,ss,st)
        sg.sp_check(op.out,pp,ee)
        sg.sg_error(out[0],rr[0],1.e-2)
        # Near cancellation, a residual-relative error magnifies FP32 FMA and
        # RMS summation rounding. Bound forward error by the two operands,
        # while keeping the projection and final normalized-output gates.
        scale=(params[phase]['alpha']*ss).norm()+(
            params[phase]['beta']*sg.sg_norm(br,1.,1.e-8)).norm()
        err=float((out[1]-rr[1]).norm()/scale.clamp_min(1.e-20))
        if err>1.e-5:raise RuntimeError(f'cancellation forward-error {m}/{phase}: {err}')
        checked+=1

out=dict(device=torch.cuda.get_device_name(),local_shared_cap=98304,checked=checked,graphs=graphs,
         max_bf16_l2=worst,elapsed_s=time.monotonic()-begin,
         source_sha256=hashlib.sha256(sg.SP_CUDA.encode()).hexdigest(),
         payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest())
(HERE/'validation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
