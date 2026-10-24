import statistics
import time

# Fixed families, compared across every seed; no per-seed winner is reported.
PP_CHOICES=((0,528),(1,528),(2,528),(3,528),(4,528),(5,528),(6,528),(7,1056),(8,1056),(9,1056))
PP_NAMES=('NT1','NT2','masked')+tuple('P'+str(i) for i in range(len(PP_CHOICES)))


def pp_control_prepare(w,raw=False):
    dev=w['w13'].device;old=rv_prepare(w)
    gs=torch.cat((w['w13_gs_gate'],w['w13_gs_up'])).float().contiguous();old['gs13']=gs
    st=dict(pre_expert=torch.ones(4096,device=dev),ones=torch.ones(32,device=dev),
        post_expert=(1+w['post_norm_w'].float()).bfloat16(),post_expert_bf16=(1+w['post_norm_w'].float()).bfloat16(),
        sms=torch.cuda.get_device_properties(dev).multi_processor_count,gs13_gate=w['w13_gs_gate'],
        gs13_up=w['w13_gs_up'],gs2=w['w2_gs'],w13_gs=gs,w2_gs=w['w2_gs'],w13p=old['w13p'],w2p=old['w2p'])
    hum=pf__Experts(dict(w,w13_gs=torch.ones(32,device=dev)),0,32)
    hum.gemm_type=pf_GemmType.GROUPED_MASKED;hum.compile([128,256]);st['humming']=hum
    for key in ('w13','w2'):assert hum.weight(key)['weight_scale'].dtype==torch.float8_e4m3fn
    if raw:
        class Raw:
            def __init__(self):self.moe_gemm=pf__grouped_module().raw17_gemm
        rr=dict(st,tiled_mod=Raw())
        for key,gated in (('w13',True),('w2',False)):
            b,sf=pf__raw_repack(w[key],w[key+'_sf'],gated);out=torch.empty_like(b)
            pf__encode_blob_fast[(b.numel()//2048,)](b.view(torch.int32),sf,out.view(torch.int32),num_warps=4)
            rr['blob13' if gated else 'blob2']=out;rr['sfb13' if gated else 'sfb2']=sf
        st['raw']=rr
    return old,st


def pp_reference(data,st,name):
    x=data['hidden_states'];m=x.shape[0];dev=x.device
    c=torch.empty(32,device=dev,dtype=torch.int32);slots=torch.empty((32,m),device=dev,dtype=torch.int32)
    pos=torch.empty((m,8),device=dev,dtype=torch.int32)
    mw_route[(32,)](data['topk_ids'],c,slots,pos,m,0,triton.next_power_of_2(m),num_warps=4)
    ids=torch.where((data['topk_ids']>=0)&(data['topk_ids']<32),data['topk_ids'],-1).int()
    args=(x,data['topk_weights'],ids,pos,c)
    if name=='raw17':return pf__raw_experts_tiled(*args,st['raw'])
    return pf__experts_humming_masked(*args,st)


def pp_measure(w,bank,old,st,ds,flush,raw=False):
    funcs=[lambda d:rv_kernel_fn(d,dict(old,nt=1)),lambda d:rv_kernel_fn(d,dict(old,nt=2)),
           lambda d:pp_reference(d,st,'masked')]
    funcs.extend(lambda d,c=choice:pipe_call(d,bank,*c) for choice in PP_CHOICES)
    names=PP_NAMES
    if raw:funcs.append(lambda d:pp_reference(d,st,'raw17'));names=names+('raw17',)
    graphs=[MWGraph(f,ds[0]) for f in funcs];wants=dc_oracles(ds,w)
    errors=[]
    for g in graphs:
        err,row=0.,0.
        for d,want in zip(ds,wants):
            g.refresh(d);got=g.replay();torch.cuda.synchronize()
            if not bool(torch.isfinite(got).all()):err=float('inf');row=float('inf');continue
            delta=got.float()-want.float();den=want.float().norm(dim=-1)
            err=max(err,float(delta.norm()/want.float().norm().clamp_min(1e-12)))
            row=max(row,float((delta.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max()))
        errors.append((err,row))
    if any(e>.01 or r>.05 for e,r in errors[:3]):raise RuntimeError('control gate failed '+str(errors[:3]))
    times=[[] for _ in names];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for i in range(8):
        for g in graphs:g.refresh(ds[i%3])
        order=list(range(len(graphs)));order=order[i%len(order):]+order[:i%len(order)]
        if i%2:order.reverse()
        for j in order:
            if errors[j][0]>.01 or errors[j][1]>.05:continue
            flush.zero_();a.record();graphs[j].replay();b.record();b.synchronize();times[j].append(a.elapsed_time(b)*1000.)
    return [(statistics.median(t) if t else float('inf'),e,r) for t,(e,r) in zip(times,errors)]


def pp_raw_phase(w,st,ds,refs,counts,flush):
    m=ds[0]['hidden_states'].shape[0];dev=w['w13'].device
    c=torch.tensor(counts,device=dev,dtype=torch.int32);ts=torch.empty(33,device=dev,dtype=torch.int32)
    ids=ds[0]['topk_ids'];slots=torch.empty((32,m),device=dev,dtype=torch.int32);pos=torch.empty_like(ids)
    mw_route[(32,)](ids,c,slots,pos,m,0,triton.next_power_of_2(m))
    pf__raw_tile_offsets_kernel[(1,)](c,ts,BNC=128)
    cap=triton.cdiv(m*8+32*127,128)*128
    xt=torch.empty(cap*4096,device=dev,dtype=torch.bfloat16);inter=torch.empty_like(xt);rows=torch.empty_like(ids)
    x=ds[0]['hidden_states'].clone()
    def gather():pf__raw_gather_tiled_kernel[(m,)](x,ids,pos,ts,xt,rows,D=4096,BNC=128,num_warps=4)
    def gemm():st['tiled_mod'].moe_gemm(st['blob13'],st['sfb13'],st['w13_gs'],xt,inter,ts,32,64,64,1,st['sms'])
    gather();gemm();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):gemm()
    starts=ts.cpu().tolist();valid=torch.tensor([r for e,n in enumerate(counts) for r in range(starts[e]*128,starts[e]*128+n)],device=dev)
    rr=valid[:,None];cc=torch.arange(4096,device=dev)[None,:]
    indices=((rr//128*64+cc//64)*128+rr%128)*64+((cc%64//8)^(rr%8))*8+cc%8
    worst=0.
    for data,rr in zip(ds,refs):
        x.copy_(data['hidden_states']);gather();g.replay();torch.cuda.synchronize()
        want=(torch.nn.functional.silu(rr[:,:4096])*rr[:,4096:]).bfloat16().float()
        worst=max(worst,float((inter[indices].float()-want).norm()/want.norm().clamp_min(1e-12)))
    if worst>.01:raise RuntimeError(f'raw17 W13/SwiGLU gate {worst}')
    vals=[];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for i in range(6):
        x.copy_(ds[i%3]['hidden_states']);gather();flush.zero_();a.record();g.replay();b.record();b.synchronize();vals.append(a.elapsed_time(b)*1000.)
    return f'RAW17_W13_SWIGLU M={m} us={statistics.median(vals):.1f} fp32={worst:.6f} gather_excluded'


def pp_w13(w,bank,ds,flush,rawst):
    # Native W13 timings, including split-K reduction, on three fresh R0 inputs.
    m=ds[0]['hidden_states'].shape[0];dev=w['w13'].device;out=[]
    plans=((0,264),(0,528),(0,1056),(1,264),(1,528),(1,1056),(2,528),
           (3,528),(4,528),(5,528),(6,528),(7,528),(7,1056),(8,1056),(9,1056))
    # Independent W13 FP32 oracle, a temporary per-expert matrix only.
    lut=torch.tensor((0.,.5,1.,1.5,2.,3.,4.,6.),device=dev)
    ids=ds[0]['topk_ids'];counts=torch.bincount(ids[ids>=0],minlength=32).cpu().tolist()
    requested={nt:sum((n+nt-1)//nt for n in counts)*8192*4096*9//16 for nt in (16,32)}
    unique=sum(n>0 for n in counts)*8192*4096*9//16
    old_tf=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
    try:
        refs=[[] for _ in ds]
        for e,n in enumerate(counts):
            if not n:continue
            q=w['w13'][e];codes=torch.stack((q&15,q>>4),-1).flatten(-2).long()
            weight=lut[codes&7]*torch.where(codes&8!=0,-1.,1.)*w['w13_sf'][e].float().repeat_interleave(16,-1)
            tok=torch.where(ids==e)[0]
            for j,data in enumerate(ds):
                want=data['hidden_states'][tok].float()@weight.t()
                want[:,:4096]*=w['w13_gs_gate'][e];want[:,4096:]*=w['w13_gs_up'][e];refs[j].append(want)
        refs=[torch.cat(x) for x in refs]
    finally:torch.backends.cuda.matmul.allow_tf32=old_tf
    for which,grid in plans:
        static={k:v.clone() if torch.is_tensor(v) else v for k,v in ds[0].items()}
        nt=PIPE_CONFIGS[which][0];cap=m*8+32*nt
        c=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32)
        order=torch.empty_like(c);tasks=torch.empty_like(off);slots=torch.empty((32,m),device=dev,dtype=torch.int32)
        pos=torch.empty_like(ids);sk=PIPE_CONFIGS[which][3]
        y=torch.empty((cap,8192),device=dev);part=torch.empty((sk,cap,8192),device=dev) if sk>1 else y
        mw_route[(32,)](ids,c,slots,pos,m,0,triton.next_power_of_2(m));mw_prefix[(1,)](c,off,order,tasks,nt)
        def launch():
            bank['mod'].pipe_run(bank['w13'],bank['w13_sf'],bank['w13_gs'],static['hidden_states'],c,off,order,tasks,slots,part,1,which,grid,m)
            if sk>1:pipe_reduce[(cap,32)](part,y,off,8192,cap,sk)
        launch();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):launch()
        offsets=off.cpu().tolist();worst=0.
        valid=torch.tensor([r for e,n in enumerate(counts) for r in range(offsets[e],offsets[e]+n)],device=dev)
        for data,want in zip(ds,refs):
            static['hidden_states'].copy_(data['hidden_states']);graph.replay();torch.cuda.synchronize()
            worst=max(worst,float((y[valid]-want).norm()/want.norm().clamp_min(1e-12)))
        if worst>.01:
            out.append(f'W13_REJECT M={m} P{which} grid={grid} fp32={worst:.6f}')
            continue
        vals=[];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        for i in range(6):
            static['hidden_states'].copy_(ds[i%3]['hidden_states']);flush.zero_();a.record();graph.replay();b.record();b.synchronize();vals.append(a.elapsed_time(b)*1000.)
        us=statistics.median(vals)
        out.append(f'W13 M={m} P{which} grid={grid} us={us:.1f} req_GBs={requested[nt]/us/1e3:.1f} unique_GBs={unique/us/1e3:.1f} fp32={worst:.6f}')
    out.append(pp_raw_phase(w,rawst,ds,refs,counts,flush))
    return out


def lab(deadline,weights,state=None):
    dev=next(v.device for v in weights.values() if torch.is_tensor(v));flush=torch.empty(128<<20,device=dev,dtype=torch.uint8)
    lines=['META larger async K stages, prefetch S-1, recycle previous stage after wait1, one CTA barrier/stage, two A register banks, gather sources hoisted, split-K FP32 reduction. Raw E4M3 stays raw; conversion in timed kernel. Same 12 seeds/3 fresh fixtures as de6a56fc with independent gate/up GS. Eight rotated cold128MiB timings. FP32 L2<=.01,row<=.05. W13 bandwidth nominal packed Q+SF, not hardware counters.']
    lines.append('ORDER '+','.join(PP_NAMES));mod=pipe_module()
    for i,cfg in enumerate(PIPE_CONFIGS):lines.append(f'CFG P{i} NT,K,stages,split,overlap={cfg} regs,localB,staticSM,dynSM,resident={mod.pipe_info(i)}')
    records=[]
    for i in range(12):
        if time.monotonic()>deadline-15:lines.append(f'INCOMPLETE before_seed={i}');break
        w=dc_weights(5353+7919*i,dev);old,control=pp_control_prepare(w,raw=i==0);bank=pipe_prepare(w)
        for m in (128,256):
            ds=dc_fixtures(w,m)
            if i==0:
                r0=dc_fixtures(w,m,True);rr=pp_measure(w,bank,old,control,r0,flush,raw=True)
                lines.append(f'R0 M={m} us='+','.join(f'{v[0]:.1f}' for v in rr)+f' raw17_last l2={max(v[1] for v in rr):.5f} row={max(v[2] for v in rr):.4f}')
                lines.extend(pp_w13(w,bank,r0,flush,control['raw']))
            vals=pp_measure(w,bank,old,control,ds,flush)
            counts=torch.bincount(ds[0]['topk_ids'][ds[0]['topk_ids']>=0],minlength=32)
            lines.append(f'S {i} M={m} P={int(counts.sum())} E={int((counts>0).sum())} us='+','.join(f'{v[0]:.1f}' for v in vals)+f' l2={max(v[1] for v in vals):.5f} row={max(v[2] for v in vals):.4f}')
            records.append((m,vals))
    for m in (128,256):
        for j,name in enumerate(PP_NAMES):
            vals=[v[j] for mm,v in records if mm==m]
            if vals:
                good=all(e<=.01 and r<=.05 for _,e,r in vals);ts=[v[0] for v in vals]
                lines.append(f'AGG {name} M={m} n={len(ts)} mean={statistics.mean(ts):.1f} med={statistics.median(ts):.1f} '+('PASS' if good else 'REJECT'))
    return lines
