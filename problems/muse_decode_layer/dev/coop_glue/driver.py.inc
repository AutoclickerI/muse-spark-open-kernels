CG_LOCAL_INSTALL=None
CG_ROWS=[]
CG_DISABLED=[]


def cg_experts(device):
    # Same stored formats and distribution as the organizer, drawn directly in
    # packed space. The proj host has no expert bank; no dequantized bank exists.
    g=torch.Generator(device=device).manual_seed(5252);w={}
    mean_sq=4.**7*math.exp(2*(math.log(2)*.4)**2)
    for name,rows,k in (('w13',8192,4096),('w2',4096,4096)):
        w[name]=torch.randint(0,256,(32,rows,k//2),device=device,dtype=torch.uint8,generator=g)
        w[name+'_sf']=torch.exp2(7+.4*torch.randn((32,rows,k//16),device=device,generator=g)).clamp(8,448).to(torch.float8_e4m3fn)
        w[name+'_gs']=torch.full((32,),1/(8.6*mean_sq*k)**.5,device=device)
    w['post_norm_w']=(.1*torch.randn(4096,device=device,generator=g)).bfloat16()
    return w


def cg_bank(m,w):
    device=w['w13'].device;i=torch.arange(8,device=device);t=torch.arange(512,device=device)
    row1=torch.cat((8*t[:,None]+i,4096+8*t[:,None]+i),1);row2=torch.arange(4096,device=device).view(256,16)
    bank=dict(w13p=torch.stack([m._repack_recoded(w['w13'][e],w['w13_sf'][e],row1) for e in range(32)]),
              w2p=torch.stack([m._repack_recoded(w['w2'][e],w['w2_sf'][e],row2) for e in range(32)]),
              gs13=w['w13_gs'],gs2=w['w2_gs'],pnw=w['post_norm_w'],
              cnt=torch.zeros(64,device=device,dtype=torch.int32),grid=torch.cuda.get_device_properties(device).multi_processor_count)
    return bank


def cg_meta_gate(meta,ids,start):
    local=(ids.long()-start+256)%256;local=torch.where(local<32,local,-1).cpu().tolist()
    m=len(local);flat=[e for row in local for e in row];expected=[0]*736
    counts=[flat.count(e) for e in range(32)];offset=[0]
    for n in counts:offset.append(offset[-1]+n)
    blocks=[(n+7)//8 for n in counts];u13=[0];u2=[0]
    for n in blocks:u13.append(u13[-1]+n*512);u2.append(u2[-1]+n*256)
    expected[:4]=[1,m,start,offset[-1]];expected[4:36]=counts;expected[36:69]=offset
    expected[69:101]=blocks;expected[101:134]=u13;expected[134:167]=u2
    expected[167:167+m]=[sum(e>=0 for e in row) for row in local]
    expected[199:455]=[-1]*256;expected[455:711]=[-1]*256
    for e in range(32):
        for pos,pair in enumerate(i for i,x in enumerate(flat) if x==e):
            expected[199+offset[e]+pos]=pair;expected[455+pair]=offset[e]+pos
    got=meta.cpu().tolist()
    if got!=expected:raise RuntimeError('route metadata v1 mismatch')


def lab(deadline,weights,host_state):
    global CG_ROWS,CG_DISABLED
    started=time.monotonic();torch.backends.cuda.matmul.allow_tf32=False
    m=dp_module('best7_cooperative_glue',BEST7_SOURCE)
    if CG_LOCAL_INSTALL is not None:CG_LOCAL_INSTALL(m)
    dev=weights['qkv_gate'].device
    experts=cg_experts(dev);synthetic=fg_weights(m,experts)
    # Both arms own the exact same original packed experts and synthetic dense.
    state=rg_setup(m,synthetic,dict(native=cg_bank(m,experts)),deadline)
    state['fused_glue_ops']={(t,p):FusedGlue(state,t,p,0) for t in (1,8,16,32,128,256) for p in (True,False,2,3)}
    legacy=dp_module('old_glue_diagnostic',LEGACY_SOURCE)
    old_op=legacy.FusedGlue(state,256,2,1)
    lines=['META cooperative glue: one RMS per row, 8 CTA chunks/row, resident cooperative grid; pool/tail/all/epi + route metadata v1',
           'META proj host: synthetic packed NVFP4 expert seed5252 (32 experts), dense seed941732; identical arms; cold ABBA6 + AA; flips disable cells']
    lines.append(f'SETUP seconds={time.monotonic()-started:.1f} remain={deadline-time.monotonic():.1f}')
    CG_ROWS=[];CG_DISABLED=[];disabled_cells=set();flush=torch.empty(256<<20,device=dev,dtype=torch.uint8)
    cases=[(b,q,w) for q in (1,8) for b in (1,16,32) for w in (2048,0)]
    for case,(b,q,window) in enumerate(cases):
        if time.monotonic()>deadline-3:
            lines.append(f'INCOMPLETE case={case}');break
        data=m._ap_generate_input(batch=b,ctx=80000,layer='sliding' if window else 'full',seed=172903,q_len=q,
            spread='agentic' if b>1 else 'fixed',dist='soft',page=1,device=dev)
        t=b*q;plan=rg_plan(m,b,q,window)
        initial=m._best7_run(data,state,plan)
        start=32*int(torch.bincount((initial[2].long()//32).flatten(),minlength=8).argmax())
        data['local_start'].fill_(start);state['_local_start']=start
        original=data['stream'].clone();a,ao=dp_graph(lambda:m._best7_run(data,state,plan));noise=dp_pair(a,a,flush)
        cells=('pool','tail','all','epi','route')
        for cell in cells:
            if (t,cell) in disabled_cells:
                lines.append(f'SKIP G{case} {cell} disabled M={t}');continue
            if time.monotonic()>deadline-2:
                lines.append(f'INCOMPLETE case={case} cell={cell}');break
            fn=(lambda:rg_run(m,data,state,plan,'route')) if cell=='route' else (lambda cell=cell:fg_run(m,data,state,plan,cell))
            bg,bo=dp_graph(fn);accepted=[]
            for seed in DP_SEEDS:
                gen=torch.Generator(device=dev).manual_seed(seed)
                data['stream'].copy_(original+torch.randn(original.shape,device=dev,generator=gen)*.2)
                data['attn_input'].copy_(m._ap_rms(data['stream']).bfloat16());data['local_start'].fill_((start+32*(seed%3))%256)
                state['_local_start']=int(data['local_start'].item());a.replay();bg.replay()
                if not torch.equal(ao[2],bo[2]):
                    product=state['fused_glue_ops'][t,True].product if cell in ('pool','all') else state['fused_glue_ops'][t,2].product if cell=='epi' else None
                    detail=fg_flip_report(m,data,state,plan,bo[2],ao[2],product)
                    row=dict(case=case,cell=cell,seed=seed,disabled=True,reason='route_flip',**detail)
                    CG_DISABLED.append(row);disabled_cells.add((t,cell));lines.append('DISABLED '+json.dumps(row,separators=(',',':')));break
                try:
                    err,_=fg_layer_gate(ao,bo,data,state)
                    if cell=='route' and t<=32:cg_meta_gate(state['route_last_meta'],bo[2],state['_local_start'])
                    fg_route_gate(m,data,plan,bo[2],state,synthetic)
                except RuntimeError as exc:
                    row=dict(case=case,cell=cell,seed=seed,disabled=True,reason=dp_reason(exc))
                    CG_DISABLED.append(row);disabled_cells.add((t,cell));lines.append('DISABLED '+json.dumps(row,separators=(',',':')));break
                pair=dp_pair(a,bg,flush)
                if not torch.equal(ao[2],bo[2]):
                    row=dict(case=case,cell=cell,seed=seed,disabled=True,reason='replay_route_flip',**fg_flip_report(m,data,state,plan,bo[2],ao[2]))
                    CG_DISABLED.append(row);disabled_cells.add((t,cell));lines.append('DISABLED '+json.dumps(row,separators=(',',':')));break
                row=dict(case=case,load=f'{b}x{q}/'+('S' if window else 'F'),cell=cell,seed=seed,error=err,flips=0,
                    noise_us=noise['b']-noise['a'],**pair)
                CG_ROWS.append(row);accepted.append(row)
            if accepted:
                lines.append(f'G{case} {cell} n={len(accepted)} A/B={statistics.mean(r["a"] for r in accepted):.2f}/{statistics.mean(r["b"] for r in accepted):.2f} '
                    f'ratio={statistics.mean(r["ratio"] for r in accepted):.4f} AA={noise["b"]-noise["a"]:.2f} err={max(r["error"] for r in accepted):.5f}')
            bg=bo=None;gc.collect()
        # Recreate the exact failed pool epilogue path and report its score gap;
        # expert values do not enter this prefix. No timing claim from this probe.
        if case==10 and time.monotonic()<deadline-2:
            gen=torch.Generator(device=dev).manual_seed(188741)
            data['stream'].copy_(original+torch.randn(original.shape,device=dev,generator=gen)*.2)
            data['attn_input'].copy_(m._ap_rms(data['stream']).bfloat16());a.replay()
            qv=m._best7_project(data['attn_input'],'qkv_gate',state,plan);at=m._best7_attention(qv,data,state,plan)
            oo=m._best7_project(at,'o',state,plan)
            sync=None
            if plan['glue']=='best5':
                raw=torch.zeros(15*t,device=dev,dtype=torch.int32)
                sync=(raw[3*t:].view(torch.float32).view(3,t,4)[0],raw[:3*t].view(3,t)[0])
            h,s=m._best7_boundary(oo,data['stream'],state,plan,'b1',sync)
            _,_,route=old_op(h,s,data['local_start'])
            detail=fg_flip_report(m,data,state,plan,route[1],ao[2],old_op.product)
            # Always include the minimum gap even when this device has no flip.
            scores=((old_op.product[:,4096:4352]+old_op.product[:,4352:4608])*.1767766952966369).sigmoid()+state['router_bias']
            top=scores.topk(9,-1).values;detail['candidate_min_gap8_9']=float((top[:,7]-top[:,8]).min())
            lines.append('LEGACY_CASE10 '+json.dumps(detail,separators=(',',':')))
        a=ao=initial=data=None;gc.collect()
    lines.append(f'END timed={len(CG_ROWS)}/180 disabled={len(CG_DISABLED)} maxL2={max((r["error"] for r in CG_ROWS),default=0):.6f} seconds={time.monotonic()-started:.1f}')
    return lines
