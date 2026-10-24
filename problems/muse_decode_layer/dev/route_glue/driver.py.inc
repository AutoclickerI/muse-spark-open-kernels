RG_LOCAL_INSTALL=None
RG_ROWS=[]
RG_MODULE=None


def rg_setup(m,weights,host_state,deadline):
    from concurrent.futures import ThreadPoolExecutor
    import hashlib
    dev=weights['w13'].device
    with ThreadPoolExecutor(max_workers=6) as pool:
        route=pool.submit(rg_module);glue=pool.submit(fg_library)
        expert=pool.submit(m._v9_recoded_module);attn=pool.submit(m._layer_attn_ext);lt=pool.submit(m._lt_module)
        hb=m._best6_humming(weights)
        native=host_state['native'];i=torch.arange(8,device=dev);t=torch.arange(512,device=dev)
        rows=torch.cat((8*t[:,None]+i,4096+8*t[:,None]+i),1)
        for name,order in (('w13',rows),('w2',torch.arange(4096,device=dev).view(256,16))):
            want=m._repack_recoded(weights[name][0],weights[name+'_sf'][0],order)
            if not torch.equal(want,native[name+'p'][0]):raise RuntimeError('host bank mismatch '+name)
        bank=dict(native,mod=expert.result(),cnt=torch.zeros_like(native['cnt']))
        original=m.v9_prepare;m.v9_prepare=lambda _:bank
        try:state=m._prepare_weights(weights)
        finally:m.v9_prepare=original
        state['lt_mod']=lt.result();attn.result();glue.result();state['route_moe_mod']=route.result()

    state['route_moe_counts']=torch.zeros_like(bank['cnt'])
    state['route_humming_count']=torch.zeros(1,device=dev,dtype=torch.int32)
    state['moe_bank']=bank;state['moe_nt_plan']={t:1 for t in (1,8,16,32,128)}
    state['moe6_adapters']=dict(humming=hb,pre_expert=state['pre_expert_w'],
        post_expert=state['post_expert_w'],ones=torch.ones(32,device=dev))
    state['proj_plan']={};state['attention_plan']={};state['prefix_plan']={};state['pdl_plan']={}
    state['lt_workspace']=torch.empty(32<<20,device=dev,dtype=torch.uint8);state['lt_algos']={}
    m._seed_v9_gemv_choices(state)
    fixed={1:('cublas',('lt',0),'custom'),8:('cublas','cublas',('lt',2)),
           16:('cublas','cublas','custom'),32:(('lt',0),('lt',7),'cublas'),
           128:('custom','cublas','cublas'),256:(('short_lt',0,0),'custom','custom')}
    for tokens,choices in fixed.items():
        tail='unpooler_slice' if tokens<=8 else 'unpooler'
        for role,choice in zip(('qkv_gate','o',tail),choices):
            state['proj_plan'][tokens,role]=choice
            if isinstance(choice,tuple) and choice[0]=='lt':
                w=state[role];x=torch.zeros((tokens,w.shape[1]),device=dev,dtype=torch.bfloat16)
                out=torch.empty((tokens,w.shape[0]),device=dev,dtype=torch.bfloat16)
                codes=state['lt_mod'].lt_options(w,x,out)
                if len(codes)<=choice[1]:raise RuntimeError(f'missing incumbent Lt {tokens}/{role}/{choice}')
                state['lt_algos'][role,tokens]=codes
        state['proj_plan'][tokens,'pool_router']=('short_lt',0,0) if tokens==128 else 'custom'
    major,minor=torch.cuda.get_device_capability();arch='90a' if major==9 else f'{major}{minor}'
    path=m.load_inline(name='muse_route_lt_'+hashlib.sha256(m.PG_RAW.encode()).hexdigest()[:12]+arch,
        cpp_sources='',cuda_sources=m.PG_RAW,functions=None,is_python_module=False,no_implicit_headers=True,
        extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}'],extra_ldflags=['-lcublasLt'],verbose=False)
    lean=dict(mod=m._PgNative(path),workspace=torch.empty(64<<20,device=dev,dtype=torch.uint8),layouts={})
    state['measured_lt']=lean
    for tokens,role in ((128,'pool_router'),(256,'qkv_gate')):
        w=state[role];n,k=w.shape
        codes,_=lean['mod'].options(tokens,n,k,0,role=='pool_router',lean['workspace'].numel())
        if len(codes)==0:raise RuntimeError('missing incumbent short Lt')
        lean['layouts'][tokens,role,0]=(w,codes)
    # Epilogue-only projections use wide token tiles at verify loads; the
    # incumbent projections remain intact for route-prologue arms.
    state['route_pool_ops']={t:FusedGlue(state,t,2,1 if t>=128 else 0,ks=1024) for t in (1,8,16,32,128,256)}
    state['route_tail_ops']={t:FusedGlue(state,t,False,1 if t>=128 else 0,ks=1024) for t in (8,128,256)}
    m._BEST5_DEADLINE=deadline
    return state


def rg_plan(m,b,q,window):
    plan=fg_control_plan(m,b,q,window)
    if q==8:plan['attention']=('layer',None)
    return plan


def lab(deadline,weights,host_state):
    global RG_ROWS,RG_MODULE
    started=time.monotonic();torch.backends.cuda.matmul.allow_tf32=False
    m=dp_module('best7_route_verify',BEST7_SOURCE);RG_MODULE=m
    if RG_LOCAL_INSTALL is not None:RG_LOCAL_INSTALL(m)
    synthetic=fg_weights(m,weights)
    for name in ('w13','w13_sf','w13_gs','w2','w2_sf','w2_gs'):synthetic[name]=synthetic[name][:32]
    state=rg_setup(m,synthetic,host_state,deadline)
    lines=['META route-only MoE prologue: incumbent pool/router projection retained; private CTA top8+norm, no grid barrier; '
           'M256 keeps Humming, folding route into its pooled/position prep',
           'META synthetic non-expert seed941732, real NVFP4 experts; best7_scale glue, c0e74bed dense dispatch; '
           '12 full layers, AB/BA6 cold256MiB, AA, 3 changed input replays, route IDs must match exactly']
    lines.append(f'SETUP seconds={time.monotonic()-started:.1f} remain={deadline-time.monotonic():.1f}')
    flush=torch.empty(256<<20,device=weights['w13'].device,dtype=torch.uint8)
    RG_ROWS=[];worst=0.;flips=0;disabled_cells=set()
    cases=[(b,q,w) for q in (1,8) for b in (1,16,32) for w in (2048,0)]
    for case,(b,q,window) in enumerate(cases):
        if time.monotonic()>deadline-3:raise RuntimeError(f'budget before requested case{case}')
        data=m._ap_generate_input(batch=b,ctx=80000,layer='sliding' if window else 'full',
            seed=172903,q_len=q,spread='agentic' if b>1 else 'fixed',dist='soft',page=1,device=flush.device)
        plan=rg_plan(m,b,q,window);t=b*q
        control=m._best7_run(data,state,plan)
        start=32*int(torch.bincount((control[2].long()//32).flatten(),minlength=8).argmax())
        data['local_start'].fill_(start);state['_local_start']=start
        original_stream=data['stream'].clone()
        a,ao=dp_graph(lambda:m._best7_run(data,state,plan));noise=dp_pair(a,a,flush)
        cells=('route','pool_epilogue','tail','both') if q==8 else ('route','pool_epilogue')
        for cell in cells:
            if (t,cell) in disabled_cells:
                lines.append(f'SKIP case={case} cell={cell} M={t}');continue
            if time.monotonic()>deadline-2:raise RuntimeError('budget before requested cell')
            bg,bo=dp_graph(lambda cell=cell:rg_run(m,data,state,plan,cell))
            for seed in DP_SEEDS:
                gen=torch.Generator(device=flush.device).manual_seed(seed)
                data['stream'].copy_(original_stream+torch.randn(original_stream.shape,device=flush.device,generator=gen)*.2)
                data['attn_input'].copy_(m._ap_rms(data['stream']).bfloat16())
                data['local_start'].fill_((start+32*(seed%3))%256);state['_local_start']=int(data['local_start'].item())
                a.replay();bg.replay()
                n=int((ao[2]!=bo[2]).any(-1).sum());flips+=n
                if n:
                    detail=fg_flip_report(m,data,state,plan,bo[2],ao[2],state['route_pool_ops'][t].product if cell=='pool_epilogue' else None)
                    RG_ROWS.append(dict(case=case,cell=cell,seed=seed,disabled=True,**detail))
                    disabled_cells.add((t,cell))
                    lines.append('DISABLED '+json.dumps(RG_ROWS[-1],separators=(',',':')))
                    break
                err,_=fg_layer_gate(ao,bo,data,state);worst=max(worst,err)
                fg_route_gate(m,data,plan,bo[2],state,synthetic)
                if not bool((state['route_moe_counts'][:34]==0).all()):raise RuntimeError('MoE counter reset')
                if not bool((state['route_humming_count']==0).all()):raise RuntimeError('Humming counter reset')
                if not all(bool((op.counts==0).all()) for op in (*state['route_pool_ops'].values(),*state['route_tail_ops'].values())):
                    raise RuntimeError('projection counter reset')
                pair=dp_pair(a,bg,flush);fg_layer_gate(ao,bo,data,state)
                if not torch.equal(ao[2],bo[2]):
                    detail=fg_flip_report(m,data,state,plan,bo[2],ao[2])
                    RG_ROWS.append(dict(case=case,cell=cell,seed=seed,disabled=True,**detail))
                    disabled_cells.add((t,cell))
                    lines.append('DISABLED '+json.dumps(RG_ROWS[-1],separators=(',',':')))
                    break
                RG_ROWS.append(dict(case=case,load=f'{b}x{q}/'+('S' if window else 'F'),cell=cell,seed=seed,
                    error=err,flips=n,noise_us=noise['b']-noise['a'],**pair))
            rr=[r for r in RG_ROWS if r['case']==case and r['cell']==cell and not r.get('disabled')];
            if not rr:
                bg=bo=None;gc.collect();continue
            lines.append(f'G{case} {cell} A/B_us={statistics.mean(r["a"] for r in rr):.2f}/{statistics.mean(r["b"] for r in rr):.2f} '
                f'B:A={statistics.mean(r["ratio"] for r in rr):.4f} AA_us={noise["b"]-noise["a"]:.2f} '
                f'err={max(r["error"] for r in rr):.5f} flips=0')
            bg=bo=None;gc.collect()
        a=ao=control=data=None;gc.collect()
    lines.append(f'END rows={len(RG_ROWS)}/108 maxL2={worst:.6f} flips={flips} '
        f'elapsed={time.monotonic()-started:.1f}s remain={deadline-time.monotonic():.1f}s')
    return lines
