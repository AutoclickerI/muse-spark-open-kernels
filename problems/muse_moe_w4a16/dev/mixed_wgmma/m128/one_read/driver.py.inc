import statistics
import time
OR_OPTS={}
OR_ROWS=[]


@functools.lru_cache(None)
def or_module():
    arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
    mod=load_inline(name='m128_one_read_'+hashlib.sha256(OR_CUDA.encode()).hexdigest()[:12],
        cpp_sources=OR_CPP,cuda_sources=OR_CUDA,functions=OR_FUNCTIONS,
        no_implicit_headers=True,with_pytorch_error_handling=False,
        extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_'+arch+',code=sm_'+arch],verbose=False)
    for prefix in ('or_rs','or_ss64','or_ss128','norig'):OR_OPTS[prefix]=getattr(mod,prefix+'_init')()
    return mod


def or_plan(data,bank,prefix='or_rs',which=3,grid=528):
    x=data['hidden_states'];m=x.shape[0];dev=x.device
    if prefix=='or_rs':nt=128
    else:
        which=(m-1)//128;nt=(which+1)*128
        if which>2:raise ValueError('expert-wide diagnostic capacity is384 tokens')
    cap=m*8+32*nt;gc=cap//nt
    c=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32)
    order=torch.empty_like(c);task=torch.empty_like(off);slots=torch.empty((32,m),device=dev,dtype=torch.int32)
    ids=data['topk_ids'].int().contiguous();tw=data['topk_weights'].float().contiguous();pos=torch.empty_like(ids)
    clear=gc*193+m;state=torch.empty(clear+m,device=dev,dtype=torch.int32)
    act=torch.empty((cap,4096),device=dev,dtype=torch.bfloat16);part=torch.empty(1,device=dev)
    y=torch.empty((cap,4096),device=dev);ss=torch.empty((cap,64),device=dev);inv=torch.empty(cap,device=dev);out=torch.empty_like(x)
    common=[c,off,order,task,slots,act];tail=[ids,tw,pos,bank['pnw'],y,ss,inv,state,out]
    v13=[bank['w13'],bank['w13_sf'],bank['w13_gs'],x]+common+[part]+tail
    v2=[bank['w2'],bank['w2_sf'],bank['w2_gs'],act]+common+[part]+tail
    rp=[v.data_ptr() for v in (ids,c,off,order,task,slots,pos,state,out)]
    p13=[v.data_ptr() for v in v13];p2=[v.data_ptr() for v in v2]
    start=data['config'].get('local_start',0);mod=or_module();owners=(v13,v2,ids,tw)
    def stream():return torch.cuda.current_stream().cuda_stream
    def route():
        _=owners
        getattr(mod,prefix+'_route')(rp,m,nt,start,clear,stream())
    def first():
        _=owners
        getattr(mod,prefix+'_run')(p13,1,which,grid,start,m,cap,stream())
    def second():
        _=owners
        getattr(mod,prefix+'_run')(p2,0,which,grid,start,m,cap,stream())
    def finish():
        _=owners
        getattr(mod,prefix+'_finish')(p2,m,start,stream());return out
    return dict(steps=(route,first,second,finish),v13=v13,v2=v2,owners=owners,out=out,NT=nt,config=which)


def or_call(data,bank,prefix='or_rs',which=3,grid=528):
    for fn in or_plan(data,bank,prefix,which,grid)['steps']:out=fn()
    return out


def or_nt_call(data,state):
    mod=or_module();x=data['hidden_states'].contiguous();m=x.shape[0];dev=x.device;stage=2
    while stage>1 and mod.norig_smem(stage,m)>OR_OPTS['norig']:stage-=1
    grid=state['grid']*mod.norig_resident(stage,m)
    inter=torch.empty((m*8,4096),device=dev,dtype=torch.bfloat16);y=torch.empty((m*8,4096),device=dev)
    ss=torch.empty(m*8,device=dev);out=torch.empty_like(x)
    values=[x,data['topk_ids'].int().contiguous(),data['topk_weights'].float().contiguous(),state['pnw'],
        state['w13p'],state['w2p'],state['gs13'],state['gs2'],inter,y,ss,state['cnt'],out]
    mod.norig_run([v.data_ptr() for v in values],m,data['config'].get('local_start',0),stage,grid,
        torch.cuda.current_stream().cuda_stream)
    return out


OR_CHOICES=(('RS64g528','or_rs',3,528),('RS128g528','or_rs',4,528),('RS64g2112','or_rs',3,2112),
    ('SS64g528','or_ss64',0,528),('SS128g528','or_ss128',0,528),('SS128g1056','or_ss128',0,1056))
OR_NAMES=('NT1rec',)+tuple(c[0] for c in OR_CHOICES)


def or_error(out,want):
    if not bool(torch.isfinite(out).all()):return float('inf'),float('inf')
    delta=out.float()-want.float();den=want.float().norm(dim=-1)
    return float(delta.norm()/want.float().norm().clamp_min(1e-12)),float((delta.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max())


def or_measure(w,bank,native,ds,flush):
    wants=dc_oracles(ds,w)
    graphs=[MWGraph(lambda d:or_nt_call(d,native),ds[0])]
    graphs += [MWGraph(lambda d,p=p,c=c,z=z:or_call(d,bank,p,c,z),ds[0]) for name,p,c,z in OR_CHOICES]
    errors=[[0.,0.] for g in graphs];rows=[];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for fi,(data,want) in enumerate(zip(ds,wants)):
        valid=[]
        for j,g in enumerate(graphs):
            g.refresh(data);out=g.replay();torch.cuda.synchronize();e,r=or_error(out,want)
            errors[j]=[max(errors[j][0],e),max(errors[j][1],r)];valid.append(e<=.01 and r<=.05)
        if not valid[0]:raise RuntimeError('actual NT1rec gate '+str(errors[0]))
        samples=[[] for g in graphs];noise=[]
        for repeat in range(6):
            order=list(range(len(graphs)))+[0];offset=repeat%len(order);order=order[offset:]+order[:offset]
            if repeat%2:order.reverse()
            aa=[]
            for j in order:
                if not valid[j]:continue
                flush.zero_();a.record();graphs[j].replay();b.record();b.synchronize();us=a.elapsed_time(b)*1000.;samples[j].append(us)
                if j==0:aa.append(us)
            noise.append(aa[1]-aa[0])
        for j,g in enumerate(graphs):
            e,r=or_error(g.out,want);errors[j]=[max(errors[j][0],e),max(errors[j][1],r)]
        ids=data['topk_ids'];start=data['config'].get('local_start',0)
        counts=torch.bincount((ids[(ids>=start)&(ids<start+32)]-start),minlength=32)
        experts=int((counts>0).sum());nt1blocks=int(torch.div(counts+7,8,rounding_mode='floor').sum())
        blocks=int(torch.div(counts+127,128,rounding_mode='floor').sum())
        if blocks!=experts:raise RuntimeError('M128 public route exceeds expert-wide128 capacity')
        rows.append(dict(fixture=fi,route=[int(counts.sum()),experts,int(counts.max())],NT1_blocks=nt1blocks,expert_blocks=blocks,
            us=[statistics.median(v) if v else float('inf') for v in samples],samples=samples,aa=noise,errors=[list(e) for e in errors]))
    return rows


def or_phases(data,bank,prefix,which,grid,flush):
    plan=or_plan(data,bank,prefix,which,grid);steps=plan['steps']
    for fn in steps:fn()
    times=[];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for phase,fn in enumerate(steps):
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):fn()
        values=[]
        for _ in range(4):
            for prep in steps[:phase]:prep()
            flush.zero_();a.record();g.replay();b.record();b.synchronize();values.append(a.elapsed_time(b)*1000.)
        times.append(statistics.median(values))
    return times


def lab(deadline,weights,state=None):
    global OR_ROWS
    OR_ROWS=[];started=time.monotonic();dev=next(v.device for v in weights.values() if torch.is_tensor(v))
    lines=['META M128 first port verified CTA-per-token W2 finalizer into repaired expert-wide RS128. SS64/128 stages one dequantized A tile and applies every expert token subgroup before overwrite; no packed weight rereads. Same uint8 recoded bank, no prepare weight expansion.',
        'META Actual NT1rec vs six expert-wide recipes;12 weight/router seeds x3 identical real routes, six reversed/rotated cold128MiB graphs/A:A; finite,FP32 L2<=.01,row<=.05 before+after. ORDER '+','.join(OR_NAMES)]
    if deadline-time.monotonic()<45:return lines+['INCOMPLETE before_compile need45s']
    mod=or_module();lines.append(f'BUILD seconds={time.monotonic()-started:.1f} left={deadline-time.monotonic():.1f}')
    for p,c in (('or_rs',3),('or_rs',4),('or_ss64',0),('or_ss128',0)):
        lines.append(f'RES {p}/{c} W13/W2={getattr(mod,p+"_info")(c,1)}/{getattr(mod,p+"_info")(c,0)}')
    flush=torch.empty(128<<20,device=dev,dtype=torch.uint8);phases=[]
    for i in range(12):
        if deadline-time.monotonic()<8:lines.append(f'INCOMPLETE before_seed={i}');break
        w=dc_weights(5353+7919*i,dev);w['w13_gs_gate']=w['w13_gs'];w['w13_gs_up']=w['w13_gs']
        bank=issue_prepare(w);native=issue_nt_prepare(w);ds=dc_fixtures(w,128)
        rows=or_measure(w,bank,native,ds,flush);OR_ROWS.extend(dict(seed=i,**r) for r in rows)
        means=[statistics.mean(r['us'][j] for r in rows) for j in range(len(OR_NAMES))]
        lines.append(f'S{i} routes='+','.join('/'.join(map(str,r['route'])) for r in rows)+' us='+','.join(f'{v:.1f}' for v in means))
        if i==0:
            for name,p,c,z in OR_CHOICES:
                vals=or_phases(ds[0],bank,p,c,z,flush)
                phases.append('PH '+name+' route/W13_act/W2_SS/final='+','.join(f'{v:.1f}' for v in vals))
        del bank,native,ds,w;gc.collect()
    lines.extend(phases)
    for j,name in enumerate(OR_NAMES):
        if OR_ROWS:
            e=max(r['errors'][j][0] for r in OR_ROWS);row=max(r['errors'][j][1] for r in OR_ROWS)
            lines.append(f'MEAN {name} n={len(OR_ROWS)}/36 us={statistics.mean(r["us"][j] for r in OR_ROWS):.2f} '
                f'vsNT1={statistics.mean(r["us"][j]/r["us"][0] for r in OR_ROWS):.4f} L2={e:.5f} row={row:.5f} '+('PASS' if e<=.01 and row<=.05 else 'REJECT'))
    if OR_ROWS:
        aa=[v for r in OR_ROWS for v in r['aa']]
        lines.append(f'READS expert_wide=1.000 NT1={statistics.mean(r["NT1_blocks"]/r["expert_blocks"] for r in OR_ROWS):.3f} nominal logical tile repeats, not DRAM counters; AA abs_mean_us={statistics.mean(abs(v) for v in aa):.3f}')
    lines.append(f'END fixtures={len(OR_ROWS)}/36 elapsed={time.monotonic()-started:.1f} left={deadline-time.monotonic():.1f}')
    return lines
