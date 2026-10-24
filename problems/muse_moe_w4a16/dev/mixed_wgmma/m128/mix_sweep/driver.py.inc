SW_OPTS={}
SW_ROWS=[]
SW_PHASES=[]


@functools.lru_cache(None)
def sw_module():
    arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
    mod=load_inline(name='m128_mix_sweep_'+hashlib.sha256(SW_CUDA.encode()).hexdigest()[:12],
        cpp_sources=SW_CPP,cuda_sources=SW_CUDA,functions=SW_FUNCTIONS,
        no_implicit_headers=True,with_pytorch_error_handling=False,
        extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_'+arch+',code=sm_'+arch],verbose=False)
    for p in ('swdist','sw_nt'):SW_OPTS[p]=getattr(mod,p+'_init')()
    return mod


def sw_choices():
    out=[];seen=set()
    def add(label,h13,n13,h2,n2):
        key=(h13,n13,h2,n2)
        if key not in seen:out.append((label,*key));seen.add(key)
    for h in (8,16,24,32):
        for n in (32,64):add(f'A{h}n{n}',h,n,16,32)
    for h in (8,16,24,32):
        for n in (32,64):add(f'B{h}n{n}',16,32,h,n)
    for h in (8,16,24,32):
        for n in (32,64):add(f'C{h}n{n}',h,n,h,n)
    add('W2_N',16,32,128,32);add('W2_D32',16,32,0,32);add('W2_D64',16,32,0,64)
    return tuple(out)


SW_CHOICES=sw_choices()
SW_NAMES=('HH_admitted',)+tuple(c[0] for c in SW_CHOICES)


def sw_plan(data,state,h13,n13,h2,n2):
    x=data['hidden_states'];m=x.shape[0];dev=x.device;cap=m*8+32*128
    base=state['base'];word=state['word'];native=state['native']
    c=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32)
    order=torch.empty_like(c);task0=torch.empty_like(off);t13=torch.empty_like(off);t2=torch.empty_like(off)
    slots=torch.empty((32,m),device=dev,dtype=torch.int32)
    ids=data['topk_ids'].int().contiguous();tw=data['topk_weights'].float().contiguous();pos=torch.empty_like(ids)
    clear=(cap//32)*193+m;counts=torch.empty(clear+m,device=dev,dtype=torch.int32)
    act=torch.empty((cap,4096),device=dev,dtype=torch.bfloat16);part=torch.empty(1,device=dev)
    y=torch.empty((cap,4096),device=dev);ss=torch.empty((cap,64),device=dev);inv=torch.empty(cap,device=dev);out=torch.empty_like(x)
    common=[c,off,order,task0,slots,act];tail=[ids,tw,pos,base['pnw'],y,ss,inv,counts,out]
    v13=[word['w13'],word['w13_sf'],word['w13_gs'],x]+common+[part]+tail;v13[7]=t13
    v2=[word['w2'],word['w2_sf'],word['w2_gs'],act]+common+[part]+tail;v2[7]=t2
    nv=[x,base['pnw'],native['w13p'],native['w2p'],base['w13_gs'],base['w2_gs'],c,off,slots,act,y,ss]
    ptrs=[[v.data_ptr() for v in row] for row in (v13,v2,nv)]
    rp=[v.data_ptr() for v in (ids,c,off,order,task0,slots,pos,counts,out)]
    mp=[v.data_ptr() for v in (c,order,t13,t2,ss)]
    mod=sw_module();start=data['config'].get('local_start',0);owners=(v13,v2,nv,task0,t13,t2,ids,tw)
    stages=2
    while stages>1 and mod.sw_nt_smem(stages,m)>SW_OPTS['sw_nt']:stages-=1
    def stream():return torch.cuda.current_stream().cuda_stream
    grids={phase:native['grid']*mod.sw_nt_resident(stages,m,phase,1) for phase in (0,1)}
    def route():
        _=owners
        mod.swdist_route(rp,m,32,start,clear,stream());mod.sw_meta(mp,cap,h13,n13,h2,n2,stream())
    def small(phase,h):
        _=owners
        if h>0:mod.sw_nt_run(ptrs[2],phase,1,m,stages,grids[phase],h,stream())
    def large(phase,h,n):
        _=owners
        if h<128:mod.swdist_run(ptrs[0 if phase else 1],phase,1 if n==32 else 2,1056,start,1,0,m,cap,stream())
    def a_small():small(1,h13)
    def a_large():large(1,h13,n13)
    def b_small():small(0,h2)
    def b_large():large(0,h2,n2)
    def first():a_small();a_large()
    def second():b_small();b_large()
    def finish():
        _=owners
        mod.swdist_finish(ptrs[1],m,start,stream());return out
    return dict(steps=(route,first,second,finish),components=(route,a_small,a_large,b_small,b_large,finish),out=out,owners=owners,v13=v13,v2=v2,tasks=(t13,t2))


def sw_call(data,state,h13,n13,h2,n2):
    for fn in sw_plan(data,state,h13,n13,h2,n2)['steps']:out=fn()
    return out


def sw_measure(w,state,ds,flush):
    wants=dc_oracles(ds,w)
    graphs=[MWGraph(lambda d:hh_experts(d['hidden_states'],d['topk_weights'],d['topk_ids'],state),ds[0])]
    graphs += [MWGraph(lambda d,a=a,n=n,b=b,k=k:sw_call(d,state,a,n,b,k),ds[0]) for name,a,n,b,k in SW_CHOICES]
    errors=[[0.,0.] for g in graphs];rows=[];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for fi,(data,want) in enumerate(zip(ds,wants)):
        for j,g in enumerate(graphs):
            g.refresh(data);out=g.replay();torch.cuda.synchronize();e,r=pm_error(out,want)
            errors[j]=[max(errors[j][0],e),max(errors[j][1],r)]
            if e>.01 or r>.05:raise RuntimeError('FP32 pre '+str((SW_NAMES[j],fi,e,r)))
        samples=[[] for g in graphs];noise=[]
        for repeat in range(6):
            order=list(range(len(graphs)))+[0];shift=repeat%len(order);order=order[shift:]+order[:shift]
            if repeat%2:order.reverse()
            aa=[]
            for j in order:
                flush.zero_();a.record();graphs[j].replay();b.record();b.synchronize();us=a.elapsed_time(b)*1000.
                samples[j].append(us)
                if j==0:aa.append(us)
            noise.append(aa[1]-aa[0])
        for j,g in enumerate(graphs):
            e,r=pm_error(g.out,want);errors[j]=[max(errors[j][0],e),max(errors[j][1],r)]
            if e>.01 or r>.05:raise RuntimeError('FP32 post '+str((SW_NAMES[j],fi,e,r)))
        ids=data['topk_ids'];counts=torch.bincount(ids[(ids>=0)&(ids<32)],minlength=32).cpu().tolist()
        rows.append(dict(fixture=fi,counts=counts,us=[statistics.median(v) for v in samples],samples=samples,errors=[list(e) for e in errors],aa=noise))
    return rows


def sw_phases(data,state,recipe,flush):
    plan=sw_plan(data,state,*recipe);steps=plan['components'];out=[]
    for f in steps:f()
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for phase,fn in enumerate(steps):
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):fn()
        samples=[]
        for _ in range(3):
            for prep in steps[:phase]:prep()
            flush.zero_();a.record();graph.replay();b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000.)
        out.append(statistics.median(samples))
    return out


def lab(deadline,weights,state=None):
    global SW_ROWS,SW_PHASES
    SW_ROWS=[];SW_PHASES=[];started=time.monotonic();dev=next(v.device for v in weights.values() if torch.is_tensor(v))
    lines=['META M128 H_H tuning: thresholds8/16/24/32; heavy WGMMA32/64 independently in W13 and W2, coupled recipes, W2-all-NT1/all-WGMMA controls. Same exact packed banks; no prepare weight expansion.',
        'META12 synthetic public MoE/router weight seeds x3 fresh identical real-route fixtures on proj host;6 cold128MiB AB/BA orders/A:A; independent FP32 before/after. ORDER '+','.join(SW_NAMES)]
    if deadline-time.monotonic()<45:return lines+['INCOMPLETE before_compile']
    m128_hh_pm_module();mod=sw_module();lines.append(f'BUILD s={time.monotonic()-started:.1f} left={deadline-time.monotonic():.1f}')
    lines.append('RES heavy32/64 W13/W2='+str([mod.swdist_info(cfg,f,1) for cfg in (1,2) for f in (1,0)]))
    flush=torch.empty(128<<20,device=dev,dtype=torch.uint8)
    # Component phases are measured on the first route of every seed. Do not
    # mistake a single seed for phase means or sum separate cold graphs into
    # whole-call latency. The normal whole-call paired graphs choose winners.
    for seed in range(12):
        if deadline-time.monotonic()<10:lines.append(f'INCOMPLETE seed={seed}');break
        w=dc_weights(5353+7919*seed,dev);w['w13_gs_gate']=w['w13_gs'];w['w13_gs_up']=w['w13_gs']
        bank=hh_prepare(w);ds=dc_fixtures(w,128);rows=sw_measure(w,bank,ds,flush);SW_ROWS.extend(dict(seed=seed,**v) for v in rows)
        for name,a,n,b,k in SW_CHOICES:
            SW_PHASES.append(dict(seed=seed,name=name,us=sw_phases(ds[0],bank,(a,n,b,k),flush)))
        lines.append('S'+str(seed)+' us='+','.join(f'{statistics.mean(r["us"][j] for r in rows):.1f}' for j in range(len(SW_NAMES))))
        del bank,w,ds;gc.collect()
    for name,a,n,b,k in SW_CHOICES:
        phase=[p['us'] for p in SW_PHASES if p['name']==name]
        if phase:lines.append('PH '+name+' r/As/Ah/Bs/Bh/f='+','.join(f'{statistics.mean(p[j] for p in phase):.1f}' for j in range(6)))
    for j,name in enumerate(SW_NAMES):
        if SW_ROWS:
            us=statistics.mean(v['us'][j] for v in SW_ROWS);ratio=statistics.mean(v['us'][j]/v['us'][0] for v in SW_ROWS)
            e=max(v['errors'][j][0] for v in SW_ROWS);r=max(v['errors'][j][1] for v in SW_ROWS)
            lines.append(f'MEAN {name} n={len(SW_ROWS)}/36 us={us:.2f} vsHH={ratio:.4f} L2={e:.5f} row={r:.5f} PASS')
    if SW_ROWS:
        noise=[a for v in SW_ROWS for a in v['aa']];lines.append(f'AA delta/abs/max_us={statistics.mean(noise):.3f}/{statistics.mean(abs(a) for a in noise):.3f}/{max(abs(a) for a in noise):.3f}')
    lines.append(f'END n={len(SW_ROWS)}/36 phase_seeds={len(set(v["seed"] for v in SW_PHASES))} lab_s={time.monotonic()-started:.1f} left={deadline-time.monotonic():.1f}')
    return lines
