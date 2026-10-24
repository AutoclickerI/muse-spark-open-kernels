import gc,json,linecache,math,statistics,time
import torch

DP_SEEDS=(172903,180822,188741)
DP_ROUNDS=6
DP_LOCAL_INSTALL=None
DP_ROWS=[]
DP_MODULES={}


class DPSpace:
    """Attribute view of an exec namespace: payloads may not register or edit modules (CONSTRAINTS 8)."""
    def __init__(self,ns):object.__setattr__(self,'_ns',ns)
    def __getattr__(self,key):
        try:return self._ns[key]
        except KeyError:raise AttributeError(key) from None
    def __setattr__(self,key,value):self._ns[key]=value


def dp_module(name,source,base=None):
    filename='<decode_paired '+name+'>'
    linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    ns=dict(base._ns) if base is not None else {'__name__':__name__,'__builtins__':__builtins__}
    exec(compile(source,filename,'exec'),ns)
    return DPSpace(ns)


def dp_reason(exc):
    return type(exc).__name__+':'+(str(exc).splitlines() or [''])[0][:85]


def dp_graph(fn):
    for _ in range(2):fn()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn()
    graph.replay()
    return graph,out


def dp_pair(a,b,flush,rounds=DP_ROUNDS):
    """Paired AB/BA rounds; flush outside events before *each* replay."""
    samples=[[],[]];ratios=[]
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    for rep in range(rounds):
        pair=[None,None]
        for index in ((0,1) if rep%2==0 else (1,0)):
            flush.zero_();start.record();(a if index==0 else b).replay();end.record();end.synchronize()
            pair[index]=start.elapsed_time(end)*1000.
            samples[index].append(pair[index])
        ratios.append(pair[1]/pair[0])
    return dict(a=statistics.mean(samples[0]),b=statistics.mean(samples[1]),
                ratio=statistics.mean(ratios),rounds=rounds,samples=samples)


def dp_error(a,b):
    af=a.float();bf=b.float();d=(af-bf).norm(dim=-1);n=bf.norm(dim=-1)
    if not bool(torch.isfinite(af).all()):return math.inf,math.inf
    return float(d.norm()/n.norm().clamp_min(1e-12)),float((d/n.clamp_min(.1*n.mean()).clamp_min(1e-12)).max())


def dp_close(actual,want,kind):
    for got,ref in zip(actual,want):
        if got.dtype in (torch.int32,torch.int64):
            if not torch.equal(got,ref):raise RuntimeError(kind+' route IDs differ')
        else:
            error,row=dp_error(got,ref)
            if error>.01 or row>.05:raise RuntimeError(f'{kind} gate {error:.5f}/{row:.5f}')


def dp_attention(m,qkv,data,state,choice,glue=None):
    prep=(glue or m).qk_prep
    if data['config']['q_len']==1:return m._layer_stage_attention(qkv,data,state)
    if choice==('layer',None):
        q,gate=prep(qkv,data,state,q_fp8=True)
        return m._attention_fa3_public(q,gate,data,state)
    # Exact FA3 split/pack calls used by best6, with the same QK preparation.
    q,gate=prep(qkv,data,state,q_fp8=True)
    return m._fa3_call(q,gate,data,choice[1])


def dp_layer(m,glue,data,state):
    t=data['config']['tokens'];device=data['stream'].device
    if glue is m:sync=[None,None,None]
    else:
        raw=(torch.empty if glue._chunks(t)==1 else torch.zeros)((15*t,),device=device,dtype=torch.int32)
        integers=raw[:3*t].view(3,t);floats=raw[3*t:].view(torch.float32).view(3,t,4)
        sync=[(floats[i],integers[i]) for i in range(3)]
    qkv=m.project(data['attn_input'],state['qkv_gate'],'qkv_gate',state)
    attn=dp_attention(m,qkv,data,state,('layer',None),glue)
    a=m.project(attn,state['o'],'o',state)
    h,s1=glue.boundary(a,data['stream'],state['alpha1'],state['beta1'],state['pre_ffn_w'],sync[0],use_pdl=True)
    pr=m.project(h,state['pool_router'],'pool_router',state,out_dtype=torch.float32)
    x,ids,wts,lids=glue.route_and_norm(pr,state,data,sync[1])
    moe=m.experts_v9(x,wts,lids,state)
    name='unpooler_slice' if t<=8 else 'unpooler'
    f=m.project(moe,state[name],name,state)
    h,s2=glue.boundary(f,s1,state['alpha2'],state['beta2'],state['next_pre_attn_w'],sync[2],
                       postw=state['post_ffn_w'],use_pdl=True)
    return h,s2,ids


def dp_fixture(m,old,data,state):
    """Actual layer prefix, then one shared routing realization for every expert."""
    qkv=m.project(data['attn_input'],state['qkv_gate'],'qkv_gate',state)
    attn=dp_attention(m,qkv,data,state,('layer',None))
    a=m.project(attn,state['o'],'o',state)
    h,s1=m.boundary(a,data['stream'],state['alpha1'],state['beta1'],state['pre_ffn_w'])
    pr=m.project(h,state['pool_router'],'pool_router',state,out_dtype=torch.float32)
    # Match the benchmark's busiest EP range for this real post-attention router.
    zero=dict(data,local_start=torch.zeros(1,device=h.device,dtype=torch.int32))
    x,ids,wts,_=m.route_and_norm(pr,state,zero)
    start=32*int(torch.bincount((ids.long()//32).flatten(),minlength=8).argmax())
    data['local_start']=torch.tensor([start],device=h.device,dtype=torch.int32)
    x,ids,wts,lids=m.route_and_norm(pr,state,data)
    # A/B comparison plus independent router oracle on identical producer values.
    t=x.shape[0];scratch=torch.zeros(15*t,device=x.device,dtype=torch.int32)
    si=scratch[:3*t].view(3,t);sf=scratch[3*t:].view(torch.float32).view(3,t,4)
    old_h=old.boundary(a,data['stream'],state['alpha1'],state['beta1'],state['pre_ffn_w'],(sf[0],si[0]))
    dp_close((h,s1),old_h,'boundary1 A/B')
    old_route=old.route_and_norm(pr,state,data,(sf[1],si[1]))
    dp_close((x,ids,wts,lids),old_route,'route A/B')
    p=pr[0].float().sum(0);scores=(p[:,4096:4352]+p[:,4352:])*m.ROUTER_MULT
    scores=scores.sigmoid();want_ids=(scores+state['router_bias']).topk(8,-1).indices.int()
    picked=scores.gather(1,want_ids.long());want_wts=picked/(picked.sum(-1,keepdim=True)+1e-15)
    want_x=(m._best5_rms(p[:,:4096])*state['pre_expert_w']).bfloat16()
    want_lids=(want_ids-start)%256;want_lids=torch.where(want_lids<32,want_lids,-1).int()
    dp_close((x,ids,wts,lids),(want_x,want_ids,want_wts,want_lids),'router FP32')
    counts=torch.bincount(lids[lids>=0],minlength=32)
    expert=dict(hidden_states=x,topk_ids=lids,topk_weights=wts,pool_partials=pr[0],pool_splits=pr[1],
                config=dict(local_count=32,d_hidden=4096,d_expert=4096,local_start=0))
    return qkv,expert,(start,int(counts.sum()),int((counts>0).sum()),int(counts.max()))


def dp_record(rows,stage,case,seed,variant,pair,**extra):
    row=dict(stage=stage,case=case,seed=seed,variant=variant,**pair,**extra)
    rows.append(row)


def dp_compact(rows,lines):
    """One line per case/variant: seed means and paired ratios survive log caps."""
    for stage in ('G','F','E'):
        groups={}
        for row in rows:
            if row['stage']==stage:groups.setdefault((row['case'],row['variant']),[]).append(row)
        for (case,variant),values in sorted(groups.items()):
            values.sort(key=lambda row:DP_SEEDS.index(row['seed']))
            seeds='/'.join(f'{v["ratio"]:.3f}' for v in values)
            a=statistics.mean(v['a'] for v in values);b=statistics.mean(v['b'] for v in values)
            noise=' N='+f'{statistics.mean(v["aa_delta"] for v in values):.2f}' if stage=='G' else ''
            lines.append(f'{stage}{case}:{variant} {seeds} {a:.1f}/{b:.1f}'+noise)


def lab(deadline,weights,host_state):
    global DP_ROWS,DP_MODULES
    started=time.monotonic();device=weights['qkv_gate'].device
    m=dp_module('best6',DP_BEST6_SOURCE);old=dp_module('best5_glue',DP_BEST5_GLUE_SOURCE,m)
    DP_MODULES={'best6':m,'old':old}
    if DP_LOCAL_INSTALL is not None:DP_LOCAL_INSTALL(m)
    for tokens,phase,setting in DP_GLUE_PLAN:
        if phase=='route':m._GLUE_ROUTE[tokens]=setting
        else:m._GLUE_BOUNDARY[tokens,phase=='b2']=setting
    # Reuse dense/router state. The board V9 uses encoded scale bytes, whereas
    # best6 NT1/NT2 consume raw E4M3: build their exact packed bank once.
    # Copy dictionaries only; no host plan or packed tensor is mutated.
    state=dict(host_state)
    state.update(proj_plan={},pdl_plan={},moe_nt_plan={},fa3_plan={})
    m._seed_v9_gemv_choices(state);m._BEST5_DEADLINE=deadline-3
    lines=['META cases=0..5:q1 B1S,F B16S,F B32S,F;6..11:q8 same; seeds=172903,180822,188741; '
           'G=whole layer old/new glue, fixed layer-attn/NT1; F=FA3 layer/split; E=NT1/expert; '
           'fields=3 seed paired-B:A ratios, seed-mean A_us/B_us; G N=A:A noise delta_us; AB/BA6 cold128MiB each replay, overhead included',
           'META F variant sp{splits}{n/t/f}=pack None/True/False; E 2=NT2 R=raw17 H=humming_masked D=dg_w8; '
           'all public ctx80000 soft/page1, agentic B>1; real post-attention busiest-EP routing; no input reuse across seeds']
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as pool:
        nt2=pool.submit(m._best5_nt2_module)
        grouped=pool.submit(m._pf_grouped_module)
        bank=pool.submit(m.v9_prepare,dict(weights,post_norm_w=weights['post_expert_norm']))
        state['moe_nt2_mod']=nt2.result()
        state['moe_bank']=bank.result()
        try:grouped.result()
        except Exception as exc:m._BEST6_GROUPED_ERROR=dp_reason(exc)
    state['moe6_adapters']=m._best6_build_adapters(weights)
    lines.append(f'SETUP seconds={time.monotonic()-started:.1f} remaining={deadline-time.monotonic():.1f}')
    for name,reason in state['moe6_adapters']['rejected'].items():lines.append('SETUP_REJECT '+name+' '+reason[:95])
    flush=torch.empty(128<<20,device=device,dtype=torch.uint8)
    rows=[];DP_ROWS=rows;completed=[];failures=[];routes=[]
    # Serving verify loads first; every stage gets all three draws before the next case.
    cases=[(b,q,w) for q in (1,8) for b in (1,16,32) for w in (2048,0)]
    for case in (11,9,7,10,8,6,5,3,1,4,2,0):
        batch,length,window=cases[case]
        for seed in DP_SEEDS:
            if time.monotonic()>deadline-5:
                lines.append(f'INCOMPLETE case={case} seed={seed}');break
            data=m._ap_generate_input(batch=batch,ctx=80000,layer='sliding' if window else 'full',
                q_len=length,spread='agentic' if batch>1 else 'fixed',dist='soft',page=1,seed=seed,device=device)
            qkv,expert,route=dp_fixture(m,old,data,state)
            routes.append((case,seed,*route))
            a,ao=dp_graph(lambda:dp_layer(m,old,data,state))
            b,bo=dp_graph(lambda:dp_layer(m,m,data,state));dp_close(bo,ao,'full layer glue')
            pair=dp_pair(a,b,flush);dp_close(bo,ao,'glue replay')
            noise=dp_pair(a,a,flush)
            dp_record(rows,'G',case,seed,'new',pair,aa_delta=noise['b']-noise['a'],aa_samples=noise['samples'])
            if length==8:
                control,co=dp_graph(lambda:dp_attention(m,qkv,data,state,('layer',None)))
                # All old FA3 settings remain available; report selected/fixed controls first.
                settings=[(32,True),(16,True),(8,None),(0,None),(32,None),(64,None)]
                for setting in settings:
                    if time.monotonic()>deadline-4:break
                    label=f'sp{setting[0]}'+{None:'n',True:'t',False:'f'}[setting[1]]
                    try:
                        cg,out=dp_graph(lambda setting=setting:dp_attention(m,qkv,data,state,('fa3',setting)))
                        dp_close((out,),(co,),'FA3 layer/split')
                        pair=dp_pair(control,cg,flush);dp_close((out,),(co,),'FA3 replay')
                        dp_record(rows,'F',case,seed,label,pair)
                    except Exception as exc:failures.append(f'REJECT F{case:02} {seed} {label} '+dp_reason(exc))
            if time.monotonic()<deadline-3:
                want=m._best5_fp32(expert,weights)
                if want is None:lines.append(f'INCOMPLETE oracle case={case} seed={seed}');break
                control,co=dp_graph(lambda:m._best6_call(expert,state,'v9nt1'))
                err,erow=m._best5_error(co,want)
                if err>.01 or erow>.05:raise RuntimeError(f'NT1 FP32 {case}/{seed}: {err}/{erow}')
                names=['v9nt2']+(['raw17','humming_masked','dg_w8'] if batch*length in (128,256) else [])
                for name in names:
                    if time.monotonic()>deadline-3:break
                    if name not in ('v9nt2',*state['moe6_adapters']['available']):continue
                    try:
                        cg,out=dp_graph(lambda name=name:m._best6_call(expert,state,name))
                        err,erow=m._best5_error(out,want)
                        if err>.01 or erow>.05:raise RuntimeError(f'FP32 {err:.5f}/{erow:.5f}')
                        pair=dp_pair(control,cg,flush)
                        err,erow=m._best5_error(out,want)
                        if err>.01 or erow>.05:raise RuntimeError(f'replay FP32 {err:.5f}/{erow:.5f}')
                        label={'v9nt2':'2','raw17':'R','humming_masked':'H','dg_w8':'D'}[name]
                        dp_record(rows,'E',case,seed,label,pair,fp32=err,worst_row=erow)
                    except Exception as exc:
                        if name=='v9nt2':raise
                        failures.append(f'REJECT E{case:02} {seed} {name} '+dp_reason(exc))
            completed.append((case,seed))
            # Graph closures are released by reassignment, not del of their captures.
            a=b=control=cg=None;ao=bo=co=out=None
            data=qkv=expert=None;gc.collect()
        if time.monotonic()>deadline-3:break
    # Use bounded compact output. Full local rows retain raw paired samples.
    lines+=failures[:12]
    lines.append('ROUTES case:localStart,pairs,experts,hot per seed '+
        ' '.join(f'{case}:'+','.join('/'.join(map(str,r[2:])) for r in routes if r[0]==case) for case in sorted({r[0] for r in routes})))
    dp_compact(rows,lines)
    errors=[r for r in rows if 'fp32' in r]
    lines.append(f'GATE experts maxL2={max((r["fp32"] for r in errors),default=0):.5f} '
                 f'maxRow={max((r["worst_row"] for r in errors),default=0):.5f}; glue IDs exact; FA3 layer/split <=1%/5%')
    lines.append(f'END fixtures={len(completed)}/36 pairs={len(rows)} rejects={len(failures)} '
                 f'elapsed={time.monotonic()-started:.1f}s remaining={deadline-time.monotonic():.1f}s')
    return lines
