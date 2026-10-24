# The control freezes receipt c0e74bed's projection/attention choices, and uses
# the current best7_scale glue table and exact recoded NT1 expert reader.
FG_LOCAL_INSTALL=None
FG_ROWS=[]
FG_MODULE=None


def fg_weights(m,experts):
    dev=experts['w13'].device;g=torch.Generator(device=dev).manual_seed(941732)
    def dense(n,k,dtype=torch.bfloat16):
        return (torch.randn((n,k),device=dev,generator=g)/k**.5).to(dtype)
    def vec(n,std=.1):return (std*torch.randn(n,device=dev,generator=g)).bfloat16()
    weights=dict(experts,qkv_gate=dense(2304,8192),o=dense(8192,1024),pooler=dense(4096,8192),
        unpooler=dense(8192,4096),attn_residual_gate=vec(8192,.3),ffn_residual_gate=vec(8192,.3),
        pre_ffn_norm=vec(8192),pre_expert_norm=vec(4096),post_ffn_norm=vec(8192),next_pre_attn_norm=vec(8192))
    weights['post_expert_norm']=experts['post_norm_w']
    mean,_=m._ap_stream_basis(str(dev))
    weights['router']=.045*torch.randn((256,8192),device=dev,generator=g)+.07*torch.randn((256,1),device=dev,generator=g)*mean
    weights['router_bias']=.055*torch.randn(256,device=dev,generator=g)
    return weights


def fg_control_plan(m,b,q,window):
    glue,attention,moe,_=m.BEST7_SETTINGS[b,q,window]
    geometry={}
    if isinstance(glue,tuple):glue,b1,route,b2=glue;geometry=dict(b1=b1,route=route,b2=b2)
    if attention=='auto':attention=('layer',None) if b==32 and window else ('attn_main',None)
    return dict(glue=glue,geometry=geometry,attention=attention,moe=moe,projection={},dense_chain=False,prefetch=0)


def fg_setup(m,weights,host_state,deadline):
    from concurrent.futures import ThreadPoolExecutor
    dev=weights['w13'].device
    with ThreadPoolExecutor(max_workers=4) as pool:
        glue=pool.submit(fg_library)
        expert=pool.submit(m._v9_recoded_module)
        attn=pool.submit(m._layer_attn_ext)
        lt=pool.submit(m._lt_module)
        # Validate the host's packed format before reusing its original NVFP4 bank.
        native=host_state['native'];i=torch.arange(8,device=dev);t=torch.arange(512,device=dev)
        rowmap=torch.cat((8*t[:,None]+i,4096+8*t[:,None]+i),1)
        for name,rows in (('w13',rowmap),('w2',torch.arange(4096,device=dev).view(256,16))):
            want=m._repack_recoded(weights[name][0],weights[name+'_sf'][0],rows)
            if not torch.equal(want,native[name+'p'][0]):raise RuntimeError('host packed bank differs: '+name)
        bank=dict(native,mod=expert.result(),cnt=torch.zeros_like(native['cnt']))
        # This is a fresh exec namespace; the host and all loaded modules stay unchanged.
        original=m.v9_prepare;m.v9_prepare=lambda _:bank
        try:state=m._prepare_weights(weights)
        finally:m.v9_prepare=original
        state['lt_mod']=lt.result();attn.result();glue.result()
    state['moe_bank']=bank;state['moe_nt_plan']={t:1 for t in (1,8,16,32)}
    state['proj_plan']={};state['attention_plan']={};state['prefix_plan']={};state['pdl_plan']={}
    state['lt_workspace']=torch.empty(32<<20,device=dev,dtype=torch.uint8);state['lt_algos']={}
    m._seed_v9_gemv_choices(state)
    fixed={1:('cublas',('lt',0),'custom'),8:('cublas','cublas',('lt',2)),
           16:('cublas','cublas','custom'),32:(('lt',0),('lt',7),'cublas')}
    for tokens,choices in fixed.items():
        tail='unpooler_slice' if tokens<=8 else 'unpooler'
        for role,choice in zip(('qkv_gate','o',tail),choices):
            state['proj_plan'][tokens,role]=choice
            if isinstance(choice,tuple):
                w=state[role];x=torch.zeros((tokens,w.shape[1]),device=dev,dtype=torch.bfloat16)
                out=torch.empty((tokens,w.shape[0]),device=dev,dtype=torch.bfloat16)
                codes=state['lt_mod'].lt_options(w,x,out)
                if len(codes)<=choice[1]:raise RuntimeError(f'missing incumbent Lt algorithm {tokens}/{role}/{choice}')
                state['lt_algos'][role,tokens]=codes
        state['proj_plan'][tokens,'pool_router']='custom'
    fg_prepare(state,recipe=0)
    m._BEST5_DEADLINE=deadline
    return state


def fg_route_gate(m,data,plan,ids,state,weights):
    qkv=m._best7_project(data['attn_input'],'qkv_gate',state,plan)
    attn=m._best7_attention(qkv,data,state,plan)
    branch=m._best7_project(attn,'o',state,plan)
    # The full-row boundary is an independent expression for old/new geometries.
    h,_=m.boundary(branch,data['stream'],state['alpha1'],state['beta1'],state['pre_ffn_w'])
    scores=(h.float()@weights['router'].T*.1767766952966369).sigmoid()+state['router_bias']
    top=scores.topk(9,-1).values;ref=scores.topk(8,-1).indices
    if not bool((scores.gather(1,ids.long())>=top[:,7:8]-.05).all()):raise RuntimeError('route score floor')
    must=top[:,:8]>top[:,8:9]+.03
    if not bool((~must|(ref[:,:,None]==ids[:,None,:]).any(-1)).all()):raise RuntimeError('must-keep route')
    local=(ids-state['_local_start']+256)%256;other=(ref-state['_local_start']+256)%256
    counts=(local<32).sum();refcounts=(other<32).sum()
    distinct=torch.unique(local[local<32]).numel();refdistinct=torch.unique(other[other<32]).numel()
    if int(counts)<float(refcounts)*.92-4 or distinct<refdistinct*.85-2:raise RuntimeError('local route count/distinct gate')


def fg_layer_gate(a,b,data,state):
    # Compare the residual update, not the much larger carried stream.
    base=state['alpha1']*state['alpha2']*data['stream']
    errors=[dp_error(b[0],a[0]),dp_error(b[1]-base,a[1]-base)]
    for error,row in errors:
        if error>.01 or row>.05:raise RuntimeError(f'whole-layer glue gate {error}/{row}')
    return max(e[0] for e in errors),int((a[2]!=b[2]).any(-1).sum())


def lab(deadline,weights,host_state):
    global FG_ROWS,FG_MODULE
    started=time.monotonic();torch.backends.cuda.matmul.allow_tf32=False
    m=dp_module('best7_scale_glue',BEST7_SOURCE);FG_MODULE=m
    if FG_LOCAL_INSTALL is not None:FG_LOCAL_INSTALL(m)
    synthetic=fg_weights(m,weights);state=fg_setup(m,synthetic,host_state,deadline)
    lines=['META best7_scale frozen; synthetic dense/norm/router seed941732, real NVFP4 experts; '
           'receipt c0e74bed projection/attention, current best7_scale glue; identical tensors per arm',
           'META B=pool(boundary1+pool/router+route),tail(unpool+boundary2),all,epi(o+boundary1,pool+route,tail); '
           'AB/BA6 cold256MiB per replay; A:A; three changed-input graph draws; FP32 hi+lo router; no spin barriers']
    lines.append(f'SETUP seconds={time.monotonic()-started:.1f} remain={deadline-time.monotonic():.1f}')
    flush=torch.empty(256<<20,device=weights['w13'].device,dtype=torch.uint8)
    FG_ROWS=[];worst=0.;total_flips=0;disabled_cells=set()
    cases=[(b,q,w) for b,q in ((1,1),(1,8),(16,1),(32,1)) for w in (2048,0)]
    for case,(b,q,window) in enumerate(cases):
        if time.monotonic()>deadline-3:
            lines.append(f'INCOMPLETE case={case}');break
        data=m._ap_generate_input(batch=b,ctx=80000,layer='sliding' if window else 'full',
            seed=172903,q_len=q,spread='agentic' if b>1 else 'fixed',dist='soft',page=1,device=flush.device)
        plan=fg_control_plan(m,b,q,window);t=b*q
        control=m._best7_run(data,state,plan)
        start=32*int(torch.bincount((control[2].long()//32).flatten(),minlength=8).argmax())
        data['local_start'].fill_(start);state['_local_start']=start
        a,ao=dp_graph(lambda:m._best7_run(data,state,plan))
        noise=dp_pair(a,a,flush)
        original_stream=data['stream'].clone()
        for cell in ('pool','tail','all','epi'):
            if (t,cell) in disabled_cells:
                lines.append(f'SKIP case={case} cell={cell} M={t}');continue
            if time.monotonic()>deadline-2:raise RuntimeError('lab budget before all glue cells')
            bg,bo=dp_graph(lambda cell=cell:fg_run(m,data,state,plan,cell))
            for seed in DP_SEEDS:
                gen=torch.Generator(device=flush.device).manual_seed(seed)
                # Same graph addresses, fresh stream/attention inputs and EP owner.
                perturb=torch.randn(data['stream'].shape,device=flush.device,generator=gen)*.2
                data['stream'].copy_(original_stream+perturb)
                data['attn_input'].copy_(m._ap_rms(data['stream']).bfloat16())
                data['local_start'].fill_((start+32*(seed%3))%256)
                state['_local_start']=int(data['local_start'].item())
                a.replay();bg.replay()
                flips=int((ao[2]!=bo[2]).any(-1).sum());total_flips+=flips
                if flips:
                    detail=fg_flip_report(m,data,state,plan,bo[2],ao[2])
                    FG_ROWS.append(dict(case=case,cell=cell,seed=seed,disabled=True,**detail))
                    disabled_cells.add((t,cell))
                    lines.append('DISABLED '+json.dumps(FG_ROWS[-1],separators=(',',':')))
                    break
                error,_=fg_layer_gate(ao,bo,data,state);worst=max(worst,error)
                # The independent FP32 router checks the published layer score/count gates.
                if cell in ('pool','all','epi'):
                    fg_route_gate(m,data,plan,bo[2],state,synthetic)
                if not all(bool((op.counts==0).all()) for op in state['fused_glue_ops'].values()):
                    raise RuntimeError('counter did not return to zero')
                pair=dp_pair(a,bg,flush)
                if not torch.equal(ao[2],bo[2]):
                    detail=fg_flip_report(m,data,state,plan,bo[2],ao[2])
                    FG_ROWS.append(dict(case=case,cell=cell,seed=seed,disabled=True,**detail))
                    disabled_cells.add((t,cell))
                    lines.append('DISABLED '+json.dumps(FG_ROWS[-1],separators=(',',':')))
                    break
                fg_layer_gate(ao,bo,data,state)
                FG_ROWS.append(dict(case=case,load=f'{b}x{q}/'+('S' if window else 'F'),cell=cell,seed=seed,
                    error=error,flips=flips,noise_us=noise['b']-noise['a'],**pair))
            rr=[r for r in FG_ROWS if r['case']==case and r['cell']==cell and not r.get('disabled')];
            if not rr:
                bg=bo=None;gc.collect();continue
            lines.append(f'G {b}x{q}{"S" if window else "F"} {cell} A/B_us='
                f'{statistics.mean(r["a"] for r in rr):.2f}/{statistics.mean(r["b"] for r in rr):.2f} '
                f'B:A={statistics.mean(r["ratio"] for r in rr):.4f} '
                f'AA_us={noise["b"]-noise["a"]:.2f} err={max(r["error"] for r in rr):.5f} flips={sum(r["flips"] for r in rr)}')
            bg=bo=None;gc.collect()
        a=ao=control=data=None;gc.collect()
    lines.append(f'END rows={len(FG_ROWS)}/96 maxL2={worst:.6f} flips={total_flips} '
        f'elapsed={time.monotonic()-started:.1f}s remain={deadline-time.monotonic():.1f}s')
    return lines
