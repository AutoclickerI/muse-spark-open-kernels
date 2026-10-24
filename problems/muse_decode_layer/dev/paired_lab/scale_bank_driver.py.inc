# These functions retain their board-host globals, including its recoded reader.
DP_HOST_CALL = v9_kernel_fn
DP_HOST_CUDA = V9_CUDA_SRC
DP_HOST_REPACK = _repack
DP_HOST_ENCODE = _encode_scales


def dp_bank_dispatch(bank,tokens):
    mod=bank['mod'];stages=2
    while stages>1 and mod.smem_needed(1,stages,tokens)>mod.smem_optin:stages-=1
    return stages,bank['grid']*mod.resident_blocks(1,stages,tokens),mod.smem_needed(1,stages,tokens)


def lab(deadline,weights,host_state):
    global DP_ROWS,DP_MODULES
    started=time.monotonic();device=weights['qkv_gate'].device
    if 'scale_e4m3' in DP_HOST_CUDA:raise RuntimeError('requires the board recoded-scale host')
    m=dp_module('best7_raw_NT1',DP_BEST6_SOURCE);DP_MODULES={'best6':m}
    if DP_LOCAL_INSTALL is not None:DP_LOCAL_INSTALL(m)
    state=dict(host_state,proj_plan={},pdl_plan={},moe_nt_plan={},fa3_plan={})
    m._seed_v9_gemv_choices(state);m._BEST5_DEADLINE=deadline-2
    # A uses best7's byte-identical reader and packer; B uses the actual host bank.
    if deadline-time.monotonic()<30:return ['INCOMPLETE insufficient setup budget']
    # Both native builds are needed before fixtures. Serial cold compilation
    # exceeded an 80 s share; overlap it, before any graph capture starts.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        bank_build=pool.submit(m.v9_prepare,dict(weights,post_norm_w=weights['post_expert_norm']))
        attention_build=pool.submit(m._layer_attn_ext)
        state['moe_bank']=bank_build.result()
        attention_build.result()
    raw=state['moe_bank'];recoded=dict(host_state['moe_bank'])
    recoded['cnt']=torch.zeros_like(recoded['cnt'])
    # Check the first expert from each bank against its own packer. Never feed
    # encoded bytes to the raw reader (the earlier small-M lab's failure).
    for name,rows in (('w13',torch.cat((8*torch.arange(512,device=device)[:,None]+torch.arange(8,device=device),
                                      4096+8*torch.arange(512,device=device)[:,None]+torch.arange(8,device=device)),1)),
                      ('w2',torch.arange(4096,device=device).view(256,16))):
        got=recoded[name+'p'][0]
        if not torch.equal(got,DP_HOST_REPACK(weights[name][0],weights[name+'_sf'][0],rows)):
            raise RuntimeError('host recoded bank mismatch '+name)
        if not torch.equal(raw[name+'p'][0],m._repack(weights[name][0],weights[name+'_sf'][0],rows)):
            raise RuntimeError('best7 raw bank mismatch '+name)
    lines=['META E scale-bank: A=best7 raw E4M3 NT1, B=board host recoded NT1; same x/IDs/weights; '
           'B:A<1 favors recode; seeds=172903,180822,188741; AB/BA6 cold128MiB before each replay',
           'META cases2,3=decode16 S,F;4,5=decode32 S,F;6,7=verify1 S,F;8,9=verify16 S,F; '
           'public ctx80000/page1/soft, agentic B>1; real full-layer prefix/busiest EP; '
           'times include equal replay overhead; oracle<=1% L2/5% row before+after; no host mutation']
    lines.append(f'SETUP seconds={time.monotonic()-started:.1f} remaining={deadline-time.monotonic():.1f}')
    flush=torch.empty(128<<20,device=device,dtype=torch.uint8)
    rows=[];DP_ROWS=rows;routes=[];completed=[]
    cases=[(b,q,w) for q in (1,8) for b in (1,16,32) for w in (2048,0)]
    for tokens in (8,16,32,128):
        da=dp_bank_dispatch(raw,tokens);db=dp_bank_dispatch(recoded,tokens)
        if da!=db:raise RuntimeError(f'dispatch differs at M{tokens}: {da}/{db}')
        lines.append(f'DISPATCH M={tokens} NT1 NB8 stages/grid/smem={da} both')
    for case in (8,9,4,5,2,3,6,7):
        batch,length,window=cases[case]
        for seed in DP_SEEDS:
            if time.monotonic()>deadline-3:
                lines.append(f'INCOMPLETE case={case} seed={seed}');break
            data=m._ap_generate_input(batch=batch,ctx=80000,layer='sliding' if window else 'full',
                q_len=length,spread='agentic' if batch>1 else 'fixed',dist='soft',page=1,seed=seed,device=device)
            qkv,expert,route=dp_fixture(m,m,data,state)
            routes.append((case,seed,*route))
            want=m._best5_fp32(expert,weights)
            if want is None:
                lines.append(f'INCOMPLETE oracle case={case} seed={seed}');break
            a,ao=dp_graph(lambda:m.v9_kernel_fn(expert,raw))
            b,bo=dp_graph(lambda:DP_HOST_CALL(expert,recoded))
            errors=[]
            for label,out in (('raw',ao),('recoded',bo)):
                error,row=m._best5_error(out,want);errors.append((error,row))
                if error>.01 or row>.05:raise RuntimeError(f'{label} FP32 {case}/{seed}: {error}/{row}')
            dp_close((bo,),(ao,),'scale banks')
            noise=dp_pair(a,a,flush)
            pair=dp_pair(a,b,flush)
            for label,out in (('raw',ao),('recoded',bo)):
                error,row=m._best5_error(out,want);errors.append((error,row))
                if error>.01 or row>.05:raise RuntimeError(f'{label} replay FP32 {case}/{seed}: {error}/{row}')
            dp_close((bo,),(ao,),'scale banks replay')
            dp_record(rows,'E',case,seed,'C',pair,aa_delta=noise['b']-noise['a'],aa_samples=noise['samples'],
                fp32=max(e[0] for e in errors),worst_row=max(e[1] for e in errors),route=route)
            completed.append((case,seed))
            # Release graph closures without deleting names they capture.
            a=b=None;ao=bo=None;out=None;data=qkv=expert=want=None;gc.collect()
        if time.monotonic()>deadline-3:break
    lines.append('ROUTES case:localStart,pairs,experts,hot per seed '+
        ' '.join(f'{case}:'+','.join('/'.join(map(str,r[2:])) for r in routes if r[0]==case)
                 for case in sorted({r[0] for r in routes})))
    dp_compact(rows,lines)
    for case in sorted({r['case'] for r in rows}):
        rr=[r for r in rows if r['case']==case]
        lines.append(f'NOISE E{case} A:A delta_us={statistics.mean(r["aa_delta"] for r in rr):.2f}')
    for tokens in (8,16,32,128):
        rr=[r for r in rows if cases[r['case']][0]*cases[r['case']][1]==tokens]
        if rr:
            a=statistics.mean(r['a'] for r in rr);b=statistics.mean(r['b'] for r in rr)
            lines.append(f'MEAN M{tokens} raw/recode_us={a:.2f}/{b:.2f} save_us={a-b:.2f} '
                         f'B:A={statistics.mean(r["ratio"] for r in rr):.4f} fixtures={len(rr)}/6')
    lines.append(f'GATE maxL2={max((r["fp32"] for r in rows),default=0):.5f} '
                 f'maxRow={max((r["worst_row"] for r in rows),default=0):.5f}; route tensors identical')
    lines.append(f'END fixtures={len(completed)}/24 pairs={len(rows)} '
                 f'elapsed={time.monotonic()-started:.1f}s remaining={deadline-time.monotonic():.1f}s')
    return lines
