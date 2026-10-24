import statistics
import time
ISSUE_OPTS={}


@functools.lru_cache(None)
def issue_module():
    arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
    mod=load_inline(name='m128_issue_'+hashlib.sha256(ISSUE_CUDA.encode()).hexdigest()[:12],
        cpp_sources=ISSUE_CPP,cuda_sources=ISSUE_CUDA,functions=ISSUE_FUNCTIONS,
        no_implicit_headers=True,with_pytorch_error_handling=False,
        extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_'+arch+',code=sm_'+arch],verbose=False)
    for p in ('ibase','iword','idist','norig','nlop','ntrace'):ISSUE_OPTS[p]=getattr(mod,p+'_init')()
    return mod


def issue_word_bank(bank):
    word=dict(bank)
    for key in ('w13_sf','w2_sf'):
        word[key]=((bank[key].to(torch.int32)<<4)+0x7000).to(torch.int16)
    return word


def issue_plan(data,bank,prefix,which=3,grid=528):
    nt=128 if prefix!='idist' else (16 if which==0 else 32)
    x=data['hidden_states'];m=x.shape[0];dev=x.device;cap=m*8+32*nt;gc=cap//nt
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
    start=data['config'].get('local_start',0);mod=issue_module();owners=(v13,v2,ids,tw)
    def stream():return torch.cuda.current_stream().cuda_stream
    def route():
        _=owners
        getattr(mod,prefix+'_route')(rp,m,nt,start,clear,stream())
    def first():
        _=owners
        if prefix=='idist':mod.idist_run(p13,1,which,grid,start,1,0,m,cap,stream())
        else:getattr(mod,prefix+'_run')(p13,1,which,grid,start,m,cap,stream())
    def second():
        _=owners
        if prefix=='idist':mod.idist_run(p2,0,which,grid,start,1,0,m,cap,stream())
        else:getattr(mod,prefix+'_run')(p2,0,which,grid,start,m,cap,stream())
        return out
    def finish():
        _=owners
        mod.idist_finish(p2,m,start,stream());return out
    steps=(route,first,second,finish) if prefix=='idist' else (route,first,second)
    return dict(steps=steps,owners=owners,out=out)


def issue_call(data,bank,prefix,which,grid):
    for f in issue_plan(data,bank,prefix,which,grid)['steps']:out=f()
    return out


def issue_nt_call(data,state,prefix='norig',trace=None):
    mod=issue_module();x=data['hidden_states'].contiguous();m=x.shape[0];dev=x.device
    stage=2;smem=getattr(mod,prefix+'_smem');opt=ISSUE_OPTS[prefix]
    while stage>1 and smem(stage,m)>opt:stage-=1
    grid=state['grid']*getattr(mod,prefix+'_resident')(stage,m)
    inter=torch.empty((m*8,4096),device=dev,dtype=torch.bfloat16);y=torch.empty((m*8,4096),device=dev)
    ss=torch.empty(m*8,device=dev);out=torch.empty_like(x)
    values=[x,data['topk_ids'].int().contiguous(),data['topk_weights'].float().contiguous(),state['pnw'],
        state['w13p'],state['w2p'],state['gs13'],state['gs2'],inter,y,ss,state['cnt'],out]
    if trace is not None:values.append(trace)
    getattr(mod,prefix+'_run')([v.data_ptr() for v in values],m,data['config'].get('local_start',0),stage,grid,
        torch.cuda.current_stream().cuda_stream)
    return out


ISSUE_CHOICES=(('F128rec','ibase',3,528),('F128word','iword',3,528),('D16word','idist',0,528),
               ('D32word','idist',1,528),('D32g','idist',1,1056))
ISSUE_NAMES=('NT1rec','NT1lop')+tuple(c[0] for c in ISSUE_CHOICES)
ISSUE_ROWS=[]


def issue_error(got,want):
    if not bool(torch.isfinite(got).all()):return float('inf'),float('inf')
    delta=got.float()-want.float();den=want.float().norm(dim=-1)
    return float(delta.norm()/want.float().norm().clamp_min(1e-12)),float((delta.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max())


def issue_measure(w,base,word,native,ds,flush):
    wants=dc_oracles(ds,w)
    graphs=[MWGraph(lambda d,p=p:issue_nt_call(d,native,p),ds[0]) for p in ('norig','nlop')]
    graphs += [MWGraph(lambda d,p=p,c=c,z=z,b=base if p=='ibase' else word:issue_call(d,b,p,c,z),ds[0])
               for name,p,c,z in ISSUE_CHOICES]
    errors=[[0.,0.] for g in graphs];records=[]
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for fi,(data,want) in enumerate(zip(ds,wants)):
        ok=[]
        for j,g in enumerate(graphs):
            g.refresh(data);out=g.replay();torch.cuda.synchronize();e,r=issue_error(out,want)
            errors[j]=[max(errors[j][0],e),max(errors[j][1],r)];ok.append(e<=.01 and r<=.05)
        if not all(ok[:2]):raise RuntimeError('NT1 gates '+str(errors[:2]))
        samples=[[] for g in graphs];noise=[]
        for repeat in range(6):
            order=list(range(len(graphs)))+[0];shift=repeat%len(order);order=order[shift:]+order[:shift]
            if repeat%2:order.reverse()
            aa=[]
            for j in order:
                if not ok[j]:continue
                flush.zero_();a.record();graphs[j].replay();b.record();b.synchronize();us=a.elapsed_time(b)*1000.
                samples[j].append(us)
                if j==0:aa.append(us)
            noise.append(aa[1]-aa[0])
        for j,g in enumerate(graphs):
            e,r=issue_error(g.out,want);errors[j]=[max(errors[j][0],e),max(errors[j][1],r)]
        counts=torch.bincount(data['topk_ids'][data['topk_ids']>=0],minlength=32)
        blocks={str(nt):int(torch.div(counts+nt-1,nt,rounding_mode='floor').sum()) for nt in (8,16,32,128)}
        records.append(dict(fixture=fi,route=[int(counts.sum()),int((counts>0).sum()),int(counts.max())],
            us=[statistics.median(v) if v else float('inf') for v in samples],samples=samples,
            errors=[list(e) for e in errors],aa=noise,token_blocks=blocks))
    return records


def issue_phases(data,bank,prefix,which,grid,flush):
    plan=issue_plan(data,bank,prefix,which,grid);steps=plan['steps']
    for f in steps:f()
    result=[];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for phase,fn in enumerate(steps):
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):fn()
        values=[]
        for _ in range(4):
            for prep in steps[:phase]:prep()
            flush.zero_();a.record();g.replay();b.record();b.synchronize();values.append(a.elapsed_time(b)*1000.)
        result.append(statistics.median(values))
    return result


def issue_nt_phases(data,native,want):
    mod=issue_module();m=data['hidden_states'].shape[0];stage=2
    while stage>1 and mod.ntrace_smem(stage,m)>ISSUE_OPTS['ntrace']:stage-=1
    grid=native['grid']*mod.ntrace_resident(stage,m)
    trace=torch.zeros((grid,5),device=data['hidden_states'].device,dtype=torch.int64)
    out=issue_nt_call(data,native,'ntrace',trace);torch.cuda.synchronize()
    e,r=issue_error(out,want)
    if e>.01 or r>.05:raise RuntimeError('NT1 phase diagnostic gate '+str((e,r)))
    vals=trace.cpu().tolist()
    # Per-CTA elapsed intervals overlap across CTAs; report distributions,
    # never sum maxima or equate them to production wall-clock phase times.
    return grid,[(min(v[j] for v in vals)/1000.,statistics.median(v[j] for v in vals)/1000.,max(v[j] for v in vals)/1000.) for j in range(5)]


def lab(deadline,weights,state=None):
    global ISSUE_ROWS
    ISSUE_ROWS=[];started=time.monotonic();dev=next(v.device for v in weights.values() if torch.is_tensor(v))
    lines=['META M128 instruction limits: actual NT1rec vs one-LOP3 blend NT1; original F128rec vs exact u16 scale-word/LOP3/wide-copy F128; D16/D32 reuse distributed W2 finalizer. No prepared expert-weight expansion.',
        'META 12 weight/router seeds x3 identical real-route fixtures; six reversed/rotated cold128MiB graphs, NT1 A:A; independent FP32 L2<=.01,row<=.05 before+after. ORDER '+','.join(ISSUE_NAMES)]
    if deadline-time.monotonic()<45:return lines+['INCOMPLETE before_compile need45s']
    mod=issue_module();lines.append(f'BUILD seconds={time.monotonic()-started:.1f} left={deadline-time.monotonic():.1f}')
    for p in ('ibase','iword'):lines.append(f'RES {p} W13/W2={getattr(mod,p+"_info")(3,1)}/{getattr(mod,p+"_info")(3,0)}')
    flush=torch.empty(128<<20,device=dev,dtype=torch.uint8)
    phase=[]
    for i in range(12):
        if deadline-time.monotonic()<9:lines.append(f'INCOMPLETE before_seed={i}');break
        w=dc_weights(5353+7919*i,dev);w['w13_gs_up']=w['w13_gs'];w['w13_gs_gate']=w['w13_gs']
        base=issue_prepare(w);word=issue_word_bank(base);native=issue_nt_prepare(w);ds=dc_fixtures(w,128)
        rows=issue_measure(w,base,word,native,ds,flush);ISSUE_ROWS.extend(dict(seed=i,**r) for r in rows)
        if i==0:
            # Separate cold graph phase medians supplement the previous lab,
            # which did not contain per-phase timers.
            for name,p,c,z in ISSUE_CHOICES:
                vals=issue_phases(ds[0],base if p=='ibase' else word,p,c,z,flush)
                phase.append('PH '+name+' route/W13_act/W2_final'+('/finish' if p=='idist' else '')+'='+','.join(f'{v:.1f}' for v in vals))
            want=dc_oracles(ds[:1],w)[0];grid,vals=issue_nt_phases(ds[0],native,want)
            phase.append('NTPH globaltimer us perCTA min/median/max route,W13_act,W2,barrier,combine grid='+str(grid)+' '+','.join('/'.join(f'{v:.1f}' for v in triple) for triple in vals))
        means=[statistics.mean(r['us'][j] for r in rows) for j in range(len(ISSUE_NAMES))]
        lines.append(f'S{i} routes='+','.join('/'.join(map(str,r['route'])) for r in rows)+' us='+','.join(f'{v:.1f}' for v in means))
        del base,word,native,ds,w;gc.collect()
    lines.extend(phase)
    for j,name in enumerate(ISSUE_NAMES):
        if ISSUE_ROWS:
            error=max(r['errors'][j][0] for r in ISSUE_ROWS);row=max(r['errors'][j][1] for r in ISSUE_ROWS)
            lines.append(f'MEAN {name} n={len(ISSUE_ROWS)}/36 us={statistics.mean(r["us"][j] for r in ISSUE_ROWS):.2f} '
                f'vsNT1={statistics.mean(r["us"][j]/r["us"][0] for r in ISSUE_ROWS):.4f} L2={error:.5f} row={row:.5f} '+('PASS' if error<=.01 and row<=.05 else 'REJECT'))
    noise=[v for r in ISSUE_ROWS for v in r['aa']]
    if ISSUE_ROWS:
        lines.append('ROUTE weight_tile_rereads NT8/16/32/128='+','.join(f'{statistics.mean(r["token_blocks"][str(nt)]/r["route"][1] for r in ISSUE_ROWS):.3f}' for nt in (8,16,32,128))+
            '; counts bound issued decode work, not measured DRAM traffic (L2 may reuse tiles)')
    if noise:lines.append(f'AA mean_delta_us={statistics.mean(noise):.3f} mean_abs_us={statistics.mean(abs(v) for v in noise):.3f} max_abs_us={max(abs(v) for v in noise):.3f}')
    lines.append(f'END fixtures={len(ISSUE_ROWS)}/36 elapsed={time.monotonic()-started:.1f} left={deadline-time.monotonic():.1f}')
    return lines
