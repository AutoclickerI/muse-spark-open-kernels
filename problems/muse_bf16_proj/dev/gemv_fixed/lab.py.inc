import gc
import linecache
import math
import statistics
import time

GF_ROWS=[]
GF_SEED=725101
GF_NAMES={'qkv_gate':'Q','o':'O','pool_router':'P','unpooler':'U','unpooler_slice':'U'}


class GFSpace:
    def __init__(self,ns):self.__dict__['_ns']=ns
    def __getattr__(self,key):
        try:return self._ns[key]
        except KeyError:raise AttributeError(key) from None
    def __setattr__(self,key,value):self._ns[key]=value


def gf_base(device):
    filename='<gemv_fixed_best7_scale>'
    linecache.cache[filename]=(len(GF_BASE_SOURCE),None,GF_BASE_SOURCE.splitlines(True),filename)
    ns={'__name__':__name__,'__builtins__':__builtins__};exec(compile(GF_BASE_SOURCE,filename,'exec'),ns)
    m=GFSpace(ns)
    def rand(name,shape,dtype=torch.float32,scale=1.):
        gen=torch.Generator(device=device).manual_seed(GF_SEED+int(_gfhash.sha256(name.encode()).hexdigest()[:8],16))
        return (scale*torch.randn(shape,device=device,generator=gen)).to(dtype)
    state={name:rand(name,shape,torch.bfloat16,shape[1]**-.5) for name,shape in
        dict(qkv_gate=(2304,8192),o=(8192,1024),pooler=(4096,8192),unpooler=(8192,4096)).items()}
    router=rand('router',(256,8192),scale=.045)
    hi=router.bfloat16();lo=(router-hi.float()).bfloat16()
    state['pool_router']=torch.cat((state.pop('pooler'),hi,lo));state['router']=router
    state['unpooler_slice']=state['unpooler'][:1024]
    for phase in (1,2):
        g=rand('gate'+str(phase),(8192,),torch.bfloat16,.3).float()/.3
        b=g.sigmoid();a=(torch.sigmoid(-g)*(1+b)).clamp_min(.001).sqrt()
        state['alpha'+str(phase)]=a;state['beta'+str(phase)]=b
    for name,n in (('pre_ffn_w',8192),('pre_expert_w',4096),('post_ffn_w',8192),('next_pre_attn_w',8192)):
        state[name]=1+rand(name,(n,),torch.bfloat16,.1).float()
    state['router_bias']=rand('router_bias',(256,),scale=.055)
    state['inv_freq']=1./(500000.**(torch.arange(0,64,2,device=device).float()/64))
    state.update(proj_plan=dict(GF_PROJ),lt_algos={},best7_stream_ops={},best7_prefetch_ops={})
    m._seed_v9_gemv_choices(state)
    state['lt_mod']=m._lt_module();state['lt_workspace']=torch.empty(32<<20,device=device,dtype=torch.uint8)
    for (t,name),choice in GF_PROJ.items():
        if isinstance(choice,tuple) and choice[0]=='lt':
            x=torch.zeros((t,state[name].shape[1]),device=device,dtype=torch.bfloat16)
            y=torch.empty((t,state[name].shape[0]),device=device,dtype=torch.bfloat16)
            codes=state['lt_mod'].lt_options(state[name],x,y)
            if codes.shape[0]<=choice[1]:raise RuntimeError('baseline Lt ordinal unavailable')
            state['lt_algos'][name,t]=codes
    return m,state


def gf_plan(m,t):
    key=(1,8,0) if t==8 else (t,1,0)
    glue,attention,moe,proj=m.BEST7_SETTINGS[key];geometry={}
    if isinstance(glue,tuple):glue,b1,route,b2=glue;geometry=dict(b1=b1,route=route,b2=b2)
    if proj!='auto' or glue!='best6':raise RuntimeError('regenerate: baseline full dense cell changed')
    return dict(glue=glue,geometry=geometry,attention=attention,moe=moe,projection={},prefetch=0,dense_chain=False)


def gf_sync(t,device):
    # Full cells use best6 glue, so this scratch is never read. Keep no extra
    # clear or allocation in that timed segment.
    return None


def gf_data(t,device):
    gen=torch.Generator(device=device).manual_seed(GF_SEED+t)
    return dict(stream=torch.randn((t,8192),device=device,generator=gen),
        local_start=torch.zeros(1,device=device,dtype=torch.int32),
        positions=torch.arange(t,device=device,dtype=torch.int64),cache_loc=torch.arange(t,device=device,dtype=torch.int64),
        k_cache=torch.empty((t,1,2,64),device=device,dtype=torch.float8_e4m3fn),
        v_cache=torch.empty((t,1,2,64),device=device,dtype=torch.float8_e4m3fn),
        k_scale=torch.ones((),device=device),v_scale=torch.ones((),device=device),
        config=dict(tokens=t,qk_scale_factor=5.4730064863875,rope_theta=0.))


def gf_value(value):
    p,s=value
    return p.sum(0) if s>1 else p[0]


def gf_check(got,want,tol,label):
    a=got.float();b=want.float();d=(a-b).norm(dim=-1);n=b.norm(dim=-1)
    err=float(d.norm()/n.norm().clamp_min(1.e-20))
    row=float((d/n.clamp_min(.1*n.mean()).clamp_min(1.e-20)).max())
    if not bool(torch.isfinite(a).all()) or err>tol or row>5*tol:
        raise RuntimeError(f'{label}: {err:.6g}/{row:.6g} bound {tol}')
    return err


def gf_graph(fn):
    for _ in range(2):fn()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn()
    graph.replay();torch.cuda.synchronize()
    return graph,out


def gf_pair(a,b,flush,rounds=8,refresh=None):
    ea=torch.cuda.Event(enable_timing=True);eb=torch.cuda.Event(enable_timing=True);vals=[[],[]]
    for rep in range(rounds):
        if refresh is not None:refresh(rep)
        for side in ((0,1) if rep%2==0 else (1,0)):
            flush.add_(1);ea.record();(a if side==0 else b).replay();eb.record();eb.synchronize()
            vals[side].append(ea.elapsed_time(eb)*1000.)
    ratios=[b/a for a,b in zip(*vals)]
    return dict(a=statistics.mean(vals[0]),b=statistics.mean(vals[1]),
        ratio=statistics.mean(ratios),seed_ratios=[statistics.mean(ratios[s::3]) for s in range(3)] if refresh else [],samples=vals)


def gf_probe(fn,stamps,flush):
    anchor=torch.empty(2,device=flush.device,dtype=torch.int64)
    graph,out=gf_graph(lambda:(gf_anchor(anchor),fn())[1])
    flush.add_(1);graph.replay();torch.cuda.synchronize()
    report=gd_summary(stamps,anchor);graph.reset();return report,out


def gf_diag_consumer(m,value,role,data,state,plan,flush):
    t=value[0].shape[1];reps=1 if role=='qkv_gate' else plan['geometry'].get('route' if role=='pool_router' else 'b1' if role=='o' else 'b2',(8,1))[1]
    stamps=torch.empty((t*reps,16),device=flush.device,dtype=torch.int64)
    fn=lambda:gd_consumer(m,value,role,data,state,plan,stamps)
    summary,got=gf_probe(fn,stamps,flush)
    want=gd_consumer(m,value,role,data,state,plan)
    for a,b in zip(got,want):
        if a.dtype in (torch.int32,torch.int64):
            if not torch.equal(a,b):raise RuntimeError('traced consumer route IDs changed')
        else:gf_check(a,b,.01 if a.dtype in (torch.bfloat16,torch.float8_e4m3fn) else .0001,'traced consumer')
    collapsed=(gf_value(value).unsqueeze(0),1)
    a,_=gf_graph(lambda:gd_consumer(m,value,role,data,state,plan))
    b,_=gf_graph(lambda:gd_consumer(m,collapsed,role,data,state,plan))
    pair=gf_pair(a,b,flush);a.reset();b.reset()
    return dict(**summary,split_vs_one=pair)


def gf_role(m,state,flush,t,role,deadline):
    w=state[role];device=w.device;plan=gf_plan(m,t);data=gf_data(t,device)
    gen=torch.Generator(device=device).manual_seed(GF_SEED+t+w.shape[0])
    banks=[torch.randn((t,w.shape[1]),device=device,dtype=torch.bfloat16,generator=gen) for _ in range(3)]
    if t>1:banks[0][::8]=0;banks[0][::8,17]=8
    x=banks[0].clone();refresh=lambda r:x.copy_(banks[r%3])
    base=lambda:m._best7_project(x,role,state,plan,out_dtype=torch.float32 if role=='pool_router' else torch.bfloat16)
    gb,bv=gf_graph(base);bc,_=gf_graph(lambda:gd_consumer(m,base(),role,data,state,plan))
    trials=[];ops={}
    # All configurations compute the same format and complete K inside a CTA.
    for cfg in GF_CONFIGS:
        if time.monotonic()>deadline-3:raise TimeoutError('role coverage incomplete')
        op=FixedProjection(w,t,cfg,fp32=role=='pool_router' or GF_PROJ[t,role]=='custom');ops[cfg]=op
        graph,out=gf_graph(lambda:op(x));consumer,_=gf_graph(lambda:gd_consumer(m,op(x),role,data,state,plan))
        err=0.
        for bank in banks:
            x.copy_(bank);graph.replay();expected=x.float()@w.float().T
            err=max(err,gf_check(gf_value(out),expected,.0001 if op.fp32 else .01,'fresh dense'))
            if role=='pool_router':
                raw=gf_value(out);gf_check(raw[:,4096:4352]+raw[:,4352:],x.float()@state['router'].T,.0001,'FP32 router')
        producer=gf_pair(gb,graph,flush,8,refresh);combined=gf_pair(bc,consumer,flush,8,refresh)
        trials.append(dict(config=cfg,error=err,producer=producer,combined=combined))
        graph.reset();consumer.reset()
    best=min(trials,key=lambda a:a['combined']['ratio']);cfg=best['config'];x.copy_(banks[0])
    finalist,fo=gf_graph(lambda:ops[cfg](x))
    fc,_=gf_graph(lambda:gd_consumer(m,ops[cfg](x),role,data,state,plan))
    final_producer=gf_pair(gb,finalist,flush,24,refresh)
    final_combined=gf_pair(bc,fc,flush,24,refresh)
    finalist.reset();fc.reset();x.copy_(banks[0])
    traced=FixedProjection(w,t,cfg,fp32=ops[cfg].fp32,trace=True)
    diagnostic,_=gf_probe(lambda:traced(x),traced.stamps,flush)
    baseline_diagnostic=None
    if GF_PROJ[t,role]=='custom':
        bn,ks,_,_=m._GEMV_CHOICES[t,*w.shape]
        stamps=torch.empty((w.shape[0]//bn*w.shape[1]//ks,16),device=device,dtype=torch.int64)
        baseline_diagnostic,copy=gf_probe(lambda:gd_project(m,x,w,stamps),stamps,flush)
        gf_check(gf_value(copy),gf_value(base()),.0001,'traced baseline GEMV')
    else:
        # Bracket opaque library kernels. This reports only the complete span,
        # never invented entry/first-load/mainloop times inside cuBLAS.
        markers=[torch.empty(2,device=device,dtype=torch.int64) for _ in range(2)]
        br,_=gf_graph(lambda:(gf_anchor(markers[0]),base(),gf_anchor(markers[1])))
        flush.add_(1);br.replay();torch.cuda.synchronize()
        baseline_diagnostic=dict(opaque_library=True,bracket_us=float(markers[1][1]-markers[0][1])/1000.)
        br.reset()
    gb.replay();torch.cuda.synchronize();cons=gf_diag_consumer(m,bv,role,data,state,plan,flush)
    row=dict(kind='role',tokens=t,role=role,baseline_choice=GF_PROJ[t,role],trials=trials,
        selected=cfg,final_producer=final_producer,final_combined=final_combined,
        diagnostic=diagnostic,baseline_diagnostic=baseline_diagnostic,consumer=cons)
    GF_ROWS.append(row);gb.reset();bc.reset();return row


def gf_segment(m,state,flush,t,choices):
    device=flush.device;plan=gf_plan(m,t);data=gf_data(t,device)
    gen=torch.Generator(device=device).manual_seed(GF_SEED+880+t)
    xs=[torch.randn((t,1024),device=device,dtype=torch.bfloat16,generator=gen) for _ in range(3)]
    streams=[torch.randn((t,8192),device=device,generator=gen) for _ in range(3)];x=xs[0].clone()
    def refresh(r):x.copy_(xs[r%3]);data['stream'].copy_(streams[r%3])
    def segment(o=None,p=None):
        a=o(x) if o else m._best7_project(x,'o',state,plan)
        h,s=m._best7_boundary(a,data['stream'],state,plan,'b1')
        pr=p(h) if p else m._best7_project(h,'pool_router',state,plan,out_dtype=torch.float32)
        return h,s,pr
    base,bv=gf_graph(segment);trials=[]
    oc=choices[t,'o'];pc=choices[t,'pool_router']
    o=FixedProjection(state['o'],t,oc,fp32=GF_PROJ[t,'o']=='custom')
    p=FixedProjection(state['pool_router'],t,pc,fp32=True)
    for tag,a,b in (('O',o,None),('P',None,p),('OP',o,p)):
        graph,out=gf_graph(lambda:segment(a,b));err=0.
        for seed in range(3):
            refresh(seed);base.replay();graph.replay()
            for got,want in zip(out[:2],bv[:2]):err=max(err,gf_check(got,want,.003,'segment boundary'))
            gf_check(gf_value(out[2]),gf_value(bv[2]),.01,'segment pool/router')
        trial=dict(arm=tag,o_config=oc,p_config=pc,error=err,**gf_pair(base,graph,flush,24,refresh))
        trials.append(trial);graph.reset()
    base.reset();row=dict(kind='segment',tokens=t,trials=trials);GF_ROWS.append(row);return row


def lab(deadline,weights,host_state=None):
    global GF_ROWS
    GF_ROWS=[];started=time.monotonic();device=next(x.device for x in weights.values() if isinstance(x,torch.Tensor))
    lines=['META best7_scale='+_gfhash.sha256(GF_BASE_SOURCE.encode()).hexdigest()[:16]+
        '; fixed dense choices from1b80ffc0; all dense/router/norm/inputs synthetic seed725101; host state unchanged; cold128MiB AB/BA; raw us producer/producer+consumer',
        'META C=clock64 cycles observer entry->firstload/wait/mainloop/local-reduce; L=prior GPU marker->first issue us (not CPU launch); W=last CTA entry->last exit,T=p95->last exit us globaltimer; B opaque=library bracket; R=consumer partial-load cycles,split vs S1 graph us; Q=qkprep; full cells; diagnostic end barrier/stamps excluded from speedups']
    old_tf=torch.backends.cuda.matmul.allow_tf32;old_bf=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    try:
        gf_library();m,state=gf_base(device);flush=torch.zeros(128<<20,device=device,dtype=torch.uint8);choices={}
        for t in (1,8,16,32):
            for role in ('qkv_gate','o','pool_router','unpooler_slice' if t<=8 else 'unpooler'):
                r=gf_role(m,state,flush,t,role,deadline);best=dict(producer=r['final_producer'],combined=r['final_combined']);choices[t,role]=r['selected']
                d=r['diagnostic'];c=r['consumer'];b=r['baseline_diagnostic'];cc=c['split_vs_one']
                bp='lib'+f'{b["bracket_us"]:.1f}' if b.get('opaque_library') else 'c'+','.join(str(round(b[k])) for k in ('entry_first_cycles','wait_cycles','main_cycles'))
                lines.append(f'R {t}{GF_NAMES[role]} B={best["producer"]["a"]:.2f}/{best["combined"]["a"]:.2f} N{r["selected"]}={best["producer"]["b"]:.2f}/{best["combined"]["b"]:.2f} '+
                    f'r={best["combined"]["ratio"]:.4f} '+
                    'C='+','.join(str(round(d[k])) for k in ('entry_first_cycles','wait_cycles','main_cycles','reduce_cycles'))+
                    f' L/W/T={d["dispatch_first_us"]:.2f}/{d["last_wave_us"]:.2f}/{d["tail_us"]:.2f} B:{bp} R={c["consumer_load_cycles"]:.0f}/{cc["a"]:.2f}:{cc["b"]:.2f} '+
                    's='+','.join(f'{s:.3f}' for s in best['combined']['seed_ratios'])+' '+
                    'v='+','.join(f'{v["combined"]["b"]:.1f}' for v in r['trials']))
            row=gf_segment(m,state,flush,t,choices)
            lines.append(f'SEG {t} '+','.join(f'{v["arm"]}={v["a"]:.2f}:{v["b"]:.2f}/{v["ratio"]:.4f}' for v in row['trials']))
            gc.collect()
        lines.append(f'END roles=16 segments=4 elapsed={time.monotonic()-started:.1f}s left={deadline-time.monotonic():.1f}s')
        return lines
    finally:
        torch.backends.cuda.matmul.allow_tf32=old_tf;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=old_bf
