import statistics
import time
ISSUE_OPTS={}
PM_ROWS=[]


@functools.lru_cache(None)
def pm_module():
    arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
    mod=load_inline(name='m128_phase_mix_'+hashlib.sha256(PM_CUDA.encode()).hexdigest()[:12],
        cpp_sources=PM_CPP,cuda_sources=PM_CUDA,functions=PM_FUNCTIONS,
        no_implicit_headers=True,with_pytorch_error_handling=False,
        extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_'+arch+',code=sm_'+arch],verbose=False)
    for p in ('idist','or_rs','norig','pm_nt'):ISSUE_OPTS[p]=getattr(mod,p+'_init')()
    return mod


def pm_plan(data,base,word,native,first='N',second='D'):
    x=data['hidden_states'];m=x.shape[0];dev=x.device;nt=32;cap=m*8+32*128
    c=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32)
    order=torch.empty_like(c);tasks=[torch.empty_like(off) for _ in range(4)]
    slots=torch.empty((32,m),device=dev,dtype=torch.int32)
    ids=data['topk_ids'].int().contiguous();tw=data['topk_weights'].float().contiguous();pos=torch.empty_like(ids)
    clear=(cap//nt)*193+m;state=torch.empty(clear+m,device=dev,dtype=torch.int32)
    act=torch.empty((cap,4096),device=dev,dtype=torch.bfloat16);part=torch.empty(1,device=dev)
    y=torch.empty((cap,4096),device=dev);ss=torch.empty((cap,64),device=dev);inv=torch.empty(cap,device=dev);out=torch.empty_like(x)
    common=[c,off,order,tasks[0],slots,act];tail=[ids,tw,pos,base['pnw'],y,ss,inv,state,out]
    def vector(bank,phase,task):
        v=[bank['w13' if phase else 'w2'],bank['w13_sf' if phase else 'w2_sf'],bank['w13_gs' if phase else 'w2_gs'],x if phase else act]+common+[part]+tail
        v[7]=task;return v
    word13=vector(word,1,tasks[0]);word2=vector(word,0,tasks[0])
    big13=vector(word,1,tasks[1]);big2=vector(word,0,tasks[1])
    rs13=vector(base,1,tasks[2]);rs2=vector(base,0,tasks[2])
    nv=[x,base['pnw'],native['w13p'],native['w2p'],base['w13_gs'],base['w2_gs'],c,off,slots,act,y,ss]
    vectors=[word13,word2,big13,big2,rs13,rs2,nv,vector(word,1,tasks[3]),vector(word,0,tasks[3])]
    ptrs=[[v.data_ptr() for v in row] for row in vectors]
    rp=[v.data_ptr() for v in (ids,c,off,order,tasks[0],slots,pos,state,out)]
    mp=[v.data_ptr() for v in (c,order,tasks[1],tasks[2],tasks[3],ss)]
    mod=pm_module();start=data['config'].get('local_start',0);owners=(vectors,ids,tw,tasks)
    stages=2
    while stages>1 and mod.pm_nt_smem(stages,m)>ISSUE_OPTS['pm_nt']:stages-=1
    def stream():return torch.cuda.current_stream().cuda_stream
    def route():
        _=owners
        mod.idist_route(rp,m,nt,start,clear,stream());mod.pm_meta(mp,cap,stream())
    def phase(backend,which):
        _=owners
        if backend in ('N','H'):
            small=int(backend=='H');grid=native['grid']*mod.pm_nt_resident(stages,m,which,small)
            mod.pm_nt_run(ptrs[6],which,small,m,stages,grid,stream())
        if backend.startswith('D') or backend=='H':
            cfg=0 if backend=='D16' else 1;grid=528 if backend=='Dlow' else 1056
            v=ptrs[(7 if which else 8) if backend=='D16' else ((2 if which else 3) if backend=='H' else (0 if which else 1))]
            mod.idist_run(v,which,cfg,grid,start,1,0,m,cap,stream())
        if backend=='R':mod.or_rs_run(ptrs[4 if which else 5],which,3,2112,start,m,cap,stream())
    def w13():phase(first,1)
    def w2():phase(second,0)
    def finish():
        _=owners
        mod.idist_finish(ptrs[1],m,start,stream());return out
    return dict(steps=(route,w13,w2,finish),out=out,owners=owners,v13=word13,v2=word2,tasks=tasks)


def pm_call(data,base,word,native,first,second):
    for fn in pm_plan(data,base,word,native,first,second)['steps']:out=fn()
    return out


# D32g and NT1rec are byte-identical admitted controls. Other rows expose each
# phase independently; H always launches both sides with disjoint GPU tasks.
PM_CHOICES=(('OneRead','R','R'),('N_D32','N','D'),('N_D16','N','D16'),('N_Dlow','N','Dlow'),
    ('H_D32','H','D'),('H_H','H','H'),('N_H','N','H'),('H_R','H','R'),('R_D32','R','D'),('N_R','N','R'),
    ('N_N','N','N'),('D16_D32','D16','D'))
PM_NAMES=('D32g','NT1rec')+tuple(x[0] for x in PM_CHOICES)


def pm_error(got,want):
    if not bool(torch.isfinite(got).all()):return float('inf'),float('inf')
    delta=got.float()-want.float();den=want.float().norm(dim=-1)
    return float(delta.norm()/want.float().norm().clamp_min(1e-12)),float((delta.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max())


def pm_measure(w,base,word,native,ds,flush):
    wants=dc_oracles(ds,w)
    graphs=[MWGraph(lambda d:issue_call(d,word,'idist',1,1056),ds[0]),MWGraph(lambda d:issue_nt_call(d,native),ds[0])]
    graphs += [MWGraph(lambda d,a=a,b=b:pm_call(d,base,word,native,a,b),ds[0]) for name,a,b in PM_CHOICES]
    errors=[[0.,0.] for g in graphs];rows=[];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for fi,(data,want) in enumerate(zip(ds,wants)):
        for j,g in enumerate(graphs):
            g.refresh(data);out=g.replay();torch.cuda.synchronize();e,r=pm_error(out,want)
            errors[j]=[max(errors[j][0],e),max(errors[j][1],r)]
            if e>.01 or r>.05:raise RuntimeError('FP32 gate '+str((PM_NAMES[j],fi,e,r)))
        samples=[[] for g in graphs];aa=[]
        for repeat in range(6):
            order=list(range(len(graphs)))+[0];shift=repeat%len(order);order=order[shift:]+order[:shift]
            if repeat%2:order.reverse()
            noise=[]
            for j in order:
                flush.zero_();a.record();graphs[j].replay();b.record();b.synchronize();us=a.elapsed_time(b)*1000.
                samples[j].append(us)
                if j==0:noise.append(us)
            aa.append(noise[1]-noise[0])
        for j,g in enumerate(graphs):
            e,r=pm_error(g.out,want);errors[j]=[max(errors[j][0],e),max(errors[j][1],r)]
            if e>.01 or r>.05:raise RuntimeError('post-timing gate '+str((PM_NAMES[j],fi,e,r)))
        ids=data['topk_ids'];start=data['config'].get('local_start',0)
        counts=torch.bincount(ids[(ids>=start)&(ids<start+32)]-start,minlength=32).cpu().tolist()
        n8=sum((n+7)//8 for n in counts);nh=sum((n+7)//8 if n<=16 else (n+31)//32 for n in counts)
        p8=n8*8;ph=sum((n+7)//8*8 if n<=16 else (n+31)//32*32 for n in counts)
        rows.append(dict(fixture=fi,counts=counts,us=[statistics.median(v) for v in samples],samples=samples,aa=aa,errors=[list(e) for e in errors],
            groups8=n8,groups_mix=nh,pad8=p8,pad_mix=ph))
    return rows


def pm_phases(data,base,word,native,first,second,flush):
    plan=pm_plan(data,base,word,native,first,second);steps=plan['steps'];values=[]
    for f in steps:f()
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for index,fn in enumerate(steps):
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):fn()
        samples=[]
        for _ in range(4):
            for prep in steps[:index]:prep()
            flush.zero_();a.record();graph.replay();b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000.)
        values.append(statistics.median(samples))
    return values


def lab(deadline,weights,state=None):
    global PM_ROWS
    PM_ROWS=[];started=time.monotonic();dev=next(v.device for v in weights.values() if torch.is_tensor(v))
    lines=['META M128 exact D32g vs NT1rec, best one-read RS64g2112, per-expert <=16 NT1/above16 WGMMA32 mix, independent W13/W2 choices. No expanded expert weights.',
        'META12 weight/router seeds x3 fresh identical fixtures; six cold128MiB AB/BA rotated graphs, D32g A:A; independent FP32 gates before/after. ORDER '+','.join(PM_NAMES)]
    if deadline-time.monotonic()<40:return lines+['INCOMPLETE before_compile need40s']
    pm_module();lines.append(f'BUILD s={time.monotonic()-started:.1f} left={deadline-time.monotonic():.1f}')
    flush=torch.empty(128<<20,device=dev,dtype=torch.uint8);phases=[]
    for seed in range(12):
        if deadline-time.monotonic()<9:lines.append(f'INCOMPLETE seed={seed}');break
        w=dc_weights(5353+7919*seed,dev);w['w13_gs_gate']=w['w13_gs'];w['w13_gs_up']=w['w13_gs']
        base=issue_prepare(w);word=issue_word_bank(base);native=issue_nt_prepare(w);ds=dc_fixtures(w,128)
        rows=pm_measure(w,base,word,native,ds,flush);PM_ROWS.extend(dict(seed=seed,**row) for row in rows)
        if seed==0:
            for name,f,s in PM_CHOICES:
                vals=pm_phases(ds[0],base,word,native,f,s,flush)
                phases.append('PH '+name+' route/W13_act/W2_SS/finish='+','.join(f'{v:.1f}' for v in vals))
        lines.append('S'+str(seed)+' us='+','.join(f'{statistics.mean(r["us"][j] for r in rows):.1f}' for j in range(len(PM_NAMES))))
        del base,word,native,w,ds;gc.collect()
    lines+=phases
    for j,name in enumerate(PM_NAMES):
        if PM_ROWS:
            us=statistics.mean(r['us'][j] for r in PM_ROWS);ratio=statistics.mean(r['us'][j]/r['us'][0] for r in PM_ROWS)
            l2=max(r['errors'][j][0] for r in PM_ROWS);err=max(r['errors'][j][1] for r in PM_ROWS)
            lines.append(f'MEAN {name} n={len(PM_ROWS)}/36 us={us:.2f} vsD32={ratio:.4f} L2={l2:.5f} row={err:.5f} PASS')
    if PM_ROWS:
        noise=[v for r in PM_ROWS for v in r['aa']];group=[1-r['groups_mix']/r['groups8'] for r in PM_ROWS];padding=[r['pad_mix']/r['pad8']-1 for r in PM_ROWS]
        lines.append('ROUTES mix group_reduction mean/min/max='+','.join(f'{f(group):.4f}' for f in (statistics.mean,min,max))+' extra_padding='+','.join(f'{f(padding):.4f}' for f in (statistics.mean,min,max)))
        lines.append(f'AA delta_mean/abs_mean/max_abs_us={statistics.mean(noise):.3f}/{statistics.mean(abs(v) for v in noise):.3f}/{max(abs(v) for v in noise):.3f}')
    lines.append(f'END n={len(PM_ROWS)}/36 lab_s={time.monotonic()-started:.1f} left={deadline-time.monotonic():.1f}')
    return lines
