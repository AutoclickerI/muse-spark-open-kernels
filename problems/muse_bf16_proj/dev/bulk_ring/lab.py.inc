BR_ROWS=[]

class BRGraph:
    def __init__(self,fn):
        self.stream=torch.cuda.Stream();self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for _ in range(2):fn()
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.out=fn()
            self.graph.replay()
        torch.cuda.current_stream().wait_stream(self.stream);torch.cuda.synchronize()
    def replay(self):self.graph.replay()
    def reset(self):self.graph.reset()

def gf_graph(fn):
    g=BRGraph(fn);return g,g.out

def br_base(weights):
    filename='<bulk_ring_best7_scale>'
    linecache.cache[filename]=(len(GF_BASE_SOURCE),None,GF_BASE_SOURCE.splitlines(True),filename)
    ns={'__name__':__name__,'__builtins__':__builtins__};exec(compile(GF_BASE_SOURCE,filename,'exec'),ns)
    m=GFSpace(ns);device=weights['o'].device
    state={key:weights[key].contiguous() for key in ('qkv_gate','o','pooler','unpooler')}
    router=weights['router'].float().contiguous();hi=router.bfloat16();lo=(router-hi.float()).bfloat16()
    state['pool_router']=torch.cat((state.pop('pooler'),hi,lo));state['router']=router
    state['unpooler_slice']=state['unpooler'][:1024]
    # The proj host supplies all learned projection weights. The boundary is
    # a declared identity-norm, fixed-gate fixture shared by every arm.
    for phase in (1,2):
        state['alpha'+str(phase)]=torch.ones(8192,device=device)
        state['beta'+str(phase)]=torch.full((8192,),.5,device=device)
    for name,n in (('pre_ffn_w',8192),('pre_expert_w',4096),('post_ffn_w',8192),('next_pre_attn_w',8192)):
        state[name]=torch.ones(n,device=device)
    state['router_bias']=torch.zeros(256,device=device)
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

def br_geometry(op):
    units=op.n//op.r*op.k//op.kc
    return dict(grid=op.grid,splits=op.splits,explicit_k_splits=op.ks,stage_bytes=op.r*op.kc*2,
        ring_bytes=op.r*op.kc*2*op.st,reserved_smem=op.reserve,
        units_min=units//op.grid,units_max=(units+op.grid-1)//op.grid)

def br_diag(op,x,flush):
    # Separate trace launch; no stamps or instrumentation in timed graphs.
    g,_=gf_graph(lambda:op(x));flush.add_(1);g.replay();torch.cuda.synchronize()
    a=op.stamps.cpu().double();dur=(a[:,1]-a[:,0])/1000.
    out=dict(ctas=len(a),sms=len(torch.unique(a[:,2])),start_spread_us=float((a[:,0].max()-a[:,0].min())/1000.),
        finish_spread_us=float((a[:,1].max()-a[:,1].min())/1000.),duration_min_us=float(dur.min()),
        duration_max_us=float(dur.max()),tail_p95_us=float((a[:,1].max()-a[:,1].quantile(.95))/1000.),
        units_min=int(a[:,3].min()),units_max=int(a[:,3].max()))
    g.reset();return out

def br_role(m,state,flush,t,role,deadline):
    w=state[role];device=w.device;plan=gf_plan(m,t);data=gf_data(t,device)
    gen=torch.Generator(device=device).manual_seed(GF_SEED+100*t+w.shape[0])
    banks=[torch.randn((t,w.shape[1]),device=device,dtype=torch.bfloat16,generator=gen) for _ in range(3)]
    column=torch.zeros_like(banks[0]);column[:,17]=8
    probes=banks+[column];expected=[bank.float()@w.float().T for bank in probes]
    x=banks[0].clone();refresh=lambda r:x.copy_(banks[r%3])
    base=lambda:m._best7_project(x,role,state,plan,out_dtype=torch.float32 if role=='pool_router' else torch.bfloat16)
    gb,bv=gf_graph(base);bc,_=gf_graph(lambda:gd_consumer(m,base(),role,data,state,plan))
    aa=gf_pair(bc,bc,flush,12,refresh);trials=[];ops={};packs={}
    for config in BR_CONFIGS:
        if time.monotonic()>deadline-8:raise TimeoutError(f'incomplete role sweep {t}/{role}/{config}')
        r,kc,st,ks,cta=BR_CONFIGS[config];packkey=(r,kc,ks)
        op=BulkRingProjection(w,t,config,packed=packs.get(packkey));packs[packkey]=op.w;ops[config]=op
        graph,out=gf_graph(lambda:op(x));cg,_=gf_graph(lambda:gd_consumer(m,op(x),role,data,state,plan))
        error=column_error=router_error=0.
        for i,(bank,want) in enumerate(zip(probes,expected)):
            x.copy_(bank);graph.replay();raw=gf_value(out)
            e=gf_check(raw,want,.0001,'bulk BF16/FP32 dot')
            error=max(error,e)
            if i==3:column_error=gf_check(raw,want,.001,'exact weight column')
            if role=='pool_router':
                router_error=max(router_error,gf_check(raw[:,4096:4352]+raw[:,4352:],bank.float()@state['router'].T,.0001,'router hi+lo'))
        producer=gf_pair(gb,graph,flush,4,refresh);combined=gf_pair(bc,cg,flush,4,refresh)
        trials.append(dict(config=config,error=error,column_error=column_error,router_error=router_error,
            producer=producer,combined=combined,geometry=br_geometry(op)))
        graph.reset();cg.reset()
    best=min(trials,key=lambda v:v['combined']['ratio']);config=best['config'];op=ops[config]
    graph,_=gf_graph(lambda:op(x));cg,_=gf_graph(lambda:gd_consumer(m,op(x),role,data,state,plan))
    producer=gf_pair(gb,graph,flush,24,refresh);combined=gf_pair(bc,cg,flush,24,refresh)
    graph.reset();cg.reset();x.copy_(banks[0])
    trace=BulkRingProjection(w,t,config,trace=True,packed=op.w)
    diag=br_diag(trace,x,flush)
    fixed=None
    if t==1 and role=='o':
        f=FixedProjection(w,1,6,fp32=False)
        graph,fo=gf_graph(lambda:f(x));cg,_=gf_graph(lambda:gd_consumer(m,f(x),role,data,state,plan))
        for bank,want in zip(probes,expected):x.copy_(bank);graph.replay();gf_check(gf_value(fo),want,.01,'unchanged O6')
        fixed=dict(producer=gf_pair(gb,graph,flush,24,refresh),combined=gf_pair(bc,cg,flush,24,refresh))
        graph.reset();cg.reset()
    row=dict(kind='role',tokens=t,role=role,selected=config,aa=aa,producer=producer,combined=combined,
        trials=trials,geometry=br_geometry(op),diagnostic=diag,fixed_o6=fixed)
    BR_ROWS.append(row);gb.reset();bc.reset();return row

def br_segment(m,state,flush,t,choices):
    plan=gf_plan(m,t);device=flush.device;data=gf_data(t,device)
    gen=torch.Generator(device=device).manual_seed(GF_SEED+880+t)
    xs=[torch.randn((t,1024),device=device,dtype=torch.bfloat16,generator=gen) for _ in range(3)]
    streams=[torch.randn((t,8192),device=device,generator=gen) for _ in range(3)];x=xs[0].clone()
    def refresh(i):x.copy_(xs[i%3]);data['stream'].copy_(streams[i%3])
    def segment(o=None,p=None):
        a=o(x) if o else m._best7_project(x,'o',state,plan)
        h,s=m._best7_boundary(a,data['stream'],state,plan,'b1')
        pr=p(h) if p else m._best7_project(h,'pool_router',state,plan,out_dtype=torch.float32)
        routes=m._best7_route(pr,state,data,plan)
        return h,s,pr,routes
    gb,bv=gf_graph(segment);aa=gf_pair(gb,gb,flush,12,refresh)
    o=BulkRingProjection(state['o'],t,choices[t,'o']);p=BulkRingProjection(state['pool_router'],t,choices[t,'pool_router'])
    arms=[('O',o,None),('P',None,p),('OP',o,p)]
    if t==1:arms.extend([('O6',FixedProjection(state['o'],1,6,fp32=False),None),
        ('O6P',FixedProjection(state['o'],1,6,fp32=False),p)])
    trials=[]
    for tag,a,b in arms:
        g,out=gf_graph(lambda:segment(a,b));error=router_error=0.;flips=0
        for i in range(3):
            refresh(i);gb.replay();g.replay()
            for got,want in zip(out[:2],bv[:2]):error=max(error,gf_check(got,want,.003,'segment boundary'))
            raw=gf_value(out[2]);want=gf_value(bv[2]);gf_check(raw,want,.01,'segment projection')
            # Isolate route arithmetic on the candidate's identical boundary
            # input; composition drift is separately measured above.
            expected=out[0].float()@state['router'].T
            router_error=max(router_error,gf_check(raw[:,4096:4352]+raw[:,4352:],expected,.0001,'segment router'))
            flips+=int((out[3][1]!=bv[3][1]).any(dim=-1).sum())
        pair=gf_pair(gb,g,flush,24,refresh)
        trials.append(dict(arm=tag,error=error,router_error=router_error,route_token_flips=flips,**pair));g.reset()
    gb.reset();row=dict(kind='segment',tokens=t,aa=aa,trials=trials);BR_ROWS.append(row);return row

def lab(deadline,weights,host_state=None):
    global BR_ROWS
    BR_ROWS=[];start=time.monotonic()
    lines=['META best7_scale42f543a0 real host BF16 dense/FP32 router; lossless packed tiles; identical inputs; identity norm/fixed gate boundary fixture; zero router bias',
        'META cold256MiB outside events; raw graph us; AB/BA three inputs AA12 finalist24; R=producer/with-consumer; S=O->boundary->pool+router->route; native ring 4/8 stages,4-32KiB/stage,split1/2/4/8,1/2CTA perSM; O6 unchanged']
    oldtf=torch.backends.cuda.matmul.allow_tf32;oldbf=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    try:
        br_library();gf_library();m,state=br_base(weights)
        if 'BR_LOCAL_INSTALL' in globals():BR_LOCAL_INSTALL(m)
        flush=torch.zeros(256<<20,device=weights['o'].device,dtype=torch.uint8);choices={}
        for t in (1,8,16,32):
            for role in ('qkv_gate','o','pool_router','unpooler_slice' if t<=8 else 'unpooler'):
                row=br_role(m,state,flush,t,role,deadline);choices[t,role]=row['selected'];p=row['producer'];c=row['combined'];g=row['geometry'];d=row['diagnostic']
                lines.append(f'R {t}{role[0].upper()} c{row["selected"]} B={p["a"]:.2f}/{c["a"]:.2f} N={p["b"]:.2f}/{c["b"]:.2f} r={c["ratio"]:.4f} s='+','.join(f'{v:.3f}' for v in c['seed_ratios'])+
                    f' aa={row["aa"]["ratio"]:.4f} G/S/B={g["grid"]}/{g["splits"]}/{g["ring_bytes"]} sm={d["sms"]} u={d["units_min"]}:{d["units_max"]} tail={d["tail_p95_us"]:.2f} v='+','.join(f'{v["combined"]["ratio"]:.3f}' for v in row['trials']))
                if row['fixed_o6']:
                    f=row['fixed_o6'];lines.append(f'O6 B={f["producer"]["a"]:.2f}/{f["combined"]["a"]:.2f} N={f["producer"]["b"]:.2f}/{f["combined"]["b"]:.2f} r={f["combined"]["ratio"]:.4f} s='+','.join(f'{v:.3f}' for v in f['combined']['seed_ratios']))
            seg=br_segment(m,state,flush,t,choices)
            lines.append(f'S {t} aa={seg["aa"]["ratio"]:.4f} '+','.join(f'{v["arm"]}={v["a"]:.2f}:{v["b"]:.2f}/{v["ratio"]:.4f}/flip{v["route_token_flips"]}' for v in seg['trials']))
            gc.collect()
        lines.append(f'END roles16 segments4 configs{len(BR_CONFIGS)} elapsed{time.monotonic()-start:.1f}s left{deadline-time.monotonic():.1f}s')
        return lines
    finally:
        torch.backends.cuda.matmul.allow_tf32=oldtf;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=oldbf
