PC_ROWS=[]
PC_SMALL=[]


@functools.lru_cache(None)
def pc_module():
    arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
    mod=load_inline(name='m128_pc_'+hashlib.sha256(PC_CUDA.encode()).hexdigest()[:12],cpp_sources=PC_CPP,cuda_sources=PC_CUDA,
        functions=['pc_init','pc_info','pc_run','pc_meta','pc_probe'],no_implicit_headers=True,with_pytorch_error_handling=False,
        extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_'+arch+',code=sm_'+arch],verbose=False)
    mod.pc_init();return mod


def pc_plan(data,state,which=1,grid=1056,mask=3,cut=16,pcut=None):
    x=data['hidden_states'];m=x.shape[0];dev=x.device
    if not 1<=m<=128:raise ValueError('producer capacity covers M1..128')
    if pcut is None:pcut=cut
    base=state['base'];native=state['native'];cap=m*8+32*128
    c=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32);order=torch.empty_like(c)
    tasks=[torch.empty_like(off) for _ in range(4)];slots=torch.empty((32,m),device=dev,dtype=torch.int32)
    ids=data['topk_ids'].int().contiguous();tw=data['topk_weights'].float().contiguous();pos=torch.empty_like(ids)
    clear=(cap//32)*193+m;counts=torch.empty(clear+m,device=dev,dtype=torch.int32)
    act=torch.empty((cap,4096),device=dev,dtype=torch.bfloat16);part=torch.empty(1,device=dev)
    y=torch.empty((cap,4096),device=dev);ss=torch.empty((cap,64),device=dev);inv=torch.empty(cap,device=dev);out=torch.empty_like(x)
    common=[c,off,order,tasks[0],slots,act];tail=[ids,tw,pos,base['pnw'],y,ss,inv,counts,out]
    def vector(bank,first,task):
        v=[bank['w13' if first else 'w2'],bank['w13_sf' if first else 'w2_sf'],bank['w13_gs' if first else 'w2_gs'],x if first else act]+common+[part]+tail
        v[7]=task;return v
    v13=vector(base,1,tasks[1]);v2=vector(base,0,tasks[1]);h13=vector(base,1,tasks[2]);h2=vector(base,0,tasks[2])
    nv=[x,base['pnw'],native['w13p'],native['w2p'],base['w13_gs'],base['w2_gs'],c,off,slots,act,y,ss]
    mid13=vector(base,1,tasks[3]);mid2=vector(base,0,tasks[3])
    ptrs=[[v.data_ptr() for v in row] for row in (v13,v2,h13,h2,nv,mid13,mid2)]
    rp=[v.data_ptr() for v in (ids,c,off,order,tasks[0],slots,pos,counts,out)]
    mp=[v.data_ptr() for v in (c,order,tasks[1],tasks[2],tasks[3],ss)]
    old=m128_hh_pm_module();mod=pc_module();start=data['config'].get('local_start',0);owners=(v13,v2,h13,h2,nv,mid13,mid2,tasks,ids,tw)
    stages=2
    while stages>1 and old.pm_nt_smem(stages,m)>M128_HH_ISSUE_OPTS['pm_nt']:stages-=1
    ngrids={f:native['grid']*old.pm_nt_resident(stages,m,f,1) for f in (0,1)}
    def stream():return torch.cuda.current_stream().cuda_stream
    def route():
        _=owners
        old.idist_route(rp,m,32,start,clear,stream());mod.pc_meta(mp,cap,cut,pcut,stream())
    def light(first):
        _=owners
        if cut:old.pm_nt_run(ptrs[4],first,1,m,stages,ngrids[first],stream())
    def heavy(first):
        _=owners
        if mask&(1 if first else 2):
            if pcut>cut:old.idist_run(ptrs[5 if first else 6],first,1,1056,start,1,0,m,cap,stream())
            mod.pc_run(ptrs[0 if first else 1],first,which,grid,m,stream())
        else:old.idist_run(ptrs[2 if first else 3],first,1,1056,start,1,0,m,cap,stream())
    def a_light():light(1)
    def a_heavy():heavy(1)
    def b_light():light(0)
    def b_heavy():heavy(0)
    def first():a_light();a_heavy()
    def second():b_light();b_heavy()
    def finish():
        _=owners
        old.idist_finish(ptrs[3],m,start,stream());return out
    return dict(steps=(route,first,second,finish),components=(route,a_light,a_heavy,b_light,b_heavy,finish),owners=owners,v13=v13,v2=v2,tasks=tasks,out=out)


def pc_call(data,state,which,grid,mask=3,cut=16,pcut=None):
    for fn in pc_plan(data,state,which,grid,mask,cut,pcut)['steps']:out=fn()
    return out


PC_CHOICES=tuple((*v,16) for v in (('HH_route',1,1056,0),('P32s2',0,1056,3),('P64s2',1,1056,3),('P128s2',2,1056,3),
    ('P64s3',3,1056,3),('P64g528',1,528,3),('P64g2112',1,2112,3),
    ('A32s2',0,1056,1),('A64s2',1,1056,1),('A128s2',2,1056,1),('A64s3',3,1056,1),
    ('A64g528',1,528,1),('A64g2112',1,2112,1),('B64s2',1,1056,2)))+(
    ('T32_P64',1,1056,3,32),('T32_A64',1,1056,1,32),('T32_P128',2,1056,3,32))
PC_NAMES=('HH_admitted',)+tuple(v[0] for v in PC_CHOICES)


def pc_measure(w,state,ds,small=False):
    wants=dc_oracles(ds,w)
    if small:
        tiny=lop3_bind(state['native'])
        graphs=[MWGraph(lambda d:lop3_experts(d['hidden_states'],d['topk_weights'],d['topk_ids'],tiny),ds[0])]
        choices=(('P64s2',1,1056,3,0),('P64s3',3,1056,3,0));names=('NT1lop',)+tuple(c[0] for c in choices)
    else:
        graphs=[MWGraph(lambda d:hh_experts(d['hidden_states'],d['topk_weights'],d['topk_ids'],state),ds[0])];choices=PC_CHOICES;names=PC_NAMES
    graphs += [MWGraph(lambda d,c=c,g=g,p=p,q=q:pc_call(d,state,c,g,p,0 if small else 16,q),ds[0]) for name,c,g,p,q in choices]
    errors=[[0.,0.] for g in graphs];records=[];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for index,(data,want) in enumerate(zip(ds,wants)):
        for j,g in enumerate(graphs):
            g.refresh(data);out=g.replay();torch.cuda.synchronize();e,r=pm_error(out,want);errors[j]=[max(e,errors[j][0]),max(r,errors[j][1])]
            if e>.01 or r>.05:raise RuntimeError('pre FP32 '+str((data['hidden_states'].shape[0],names[j],index,e,r)))
        samples=[[] for g in graphs];noise=[]
        for repeat in range(6):
            order=list(range(len(graphs)))+[0];shift=repeat%len(order);order=order[shift:]+order[:shift]
            if repeat%2:order.reverse()
            aa=[]
            for j in order:
                lab_cold_l2();a.record();graphs[j].replay();b.record();b.synchronize();us=a.elapsed_time(b)*1000.
                samples[j].append(us)
                if j==0:aa.append(us)
            noise.append(aa[1]-aa[0])
        for j,g in enumerate(graphs):
            e,r=pm_error(g.out,want);errors[j]=[max(e,errors[j][0]),max(r,errors[j][1])]
            if e>.01 or r>.05:raise RuntimeError('post FP32 '+str((names[j],index,e,r)))
        ids=data['topk_ids'];counts=torch.bincount(ids[(ids>=0)&(ids<32)],minlength=32).cpu().tolist()
        records.append(dict(M=data['hidden_states'].shape[0],fixture=index,names=names,us=[statistics.median(v) for v in samples],samples=samples,errors=[list(v) for v in errors],aa=noise,counts=counts,
            heavy_experts=sum(c>16 for c in counts),HH_heavy_groups=sum((c+31)//32 for c in counts if c>16)))
    return records


def pc_phases(data,state,which,grid,mask,pcut):
    plan=pc_plan(data,state,which,grid,mask,16,pcut);steps=plan['components'];result=[]
    for f in steps:f()
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for index,fn in enumerate(steps):
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):fn()
        samples=[]
        for _ in range(3):
            for prep in steps[:index]:prep()
            lab_cold_l2();a.record();graph.replay();b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000.)
        result.append(statistics.median(samples))
    return result


def lab(deadline,weights,state=None):
    global PC_ROWS,PC_SMALL
    PC_ROWS=[];PC_SMALL=[];started=time.monotonic();dev=next(v.device for v in weights.values() if torch.is_tensor(v))
    lines=['META corrected PC:128 producer/128 WGMMA consumer threads,2/3-slot full/empty mbarriers; each expert/channel/K64 packed tile decoded once, shared by every N subgroup; exact u8 recode; no prepared expanded expert weights.',
        'META12 public synthetic MoE/router seeds x3 identical routes; M128 vs exact S7 H_H (u8 codes), M16/32 vs NT1lop;6 cold AB/BA (judge-style read: lab_cold_l2),A:A,independent FP32 before/after. ORDER '+','.join(PC_NAMES)]
    if deadline-time.monotonic()<50:return lines+['INCOMPLETE before_compile']
    m128_hh_pm_module();mod=pc_module();nt1_module('lop3')
    lines.append(f'BUILD s={time.monotonic()-started:.1f} left={deadline-time.monotonic():.1f}')
    for cfg in (0,1,2,3):lines.append(f'RES cfg{cfg} M128 W13/W2='+str([mod.pc_info(cfg,128,f) for f in (1,0)]))
    phases=[]
    for seed in range(12):
        if deadline-time.monotonic()<12:lines.append(f'INCOMPLETE seed={seed}');break
        w=dc_weights(5353+7919*seed,dev);w['w13_gs_gate']=w['w13_gs'];w['w13_gs_up']=w['w13_gs']
        bank=hh_prepare(w);ds=dc_fixtures(w,128);rows=pc_measure(w,bank,ds);PC_ROWS.extend(dict(seed=seed,**v) for v in rows)
        if seed==0:
            for name,c,g,p,q in PC_CHOICES:
                if p==3 or p==0:phases.append('PH '+name+' r/Al/Ah/Bl/Bh/f='+','.join(f'{v:.1f}' for v in pc_phases(ds[0],bank,c,g,p,q)))
        for M in (16,32):PC_SMALL.extend(dict(seed=seed,**v) for v in pc_measure(w,bank,dc_fixtures(w,M),True))
        lines.append('S'+str(seed)+' us='+','.join(f'{statistics.mean(v["us"][j] for v in rows):.1f}' for j in range(len(PC_NAMES))))
        del bank,w,ds;gc.collect()
    lines+=phases
    for j,name in enumerate(PC_NAMES):
        if PC_ROWS:
            mean=statistics.mean(v['us'][j] for v in PC_ROWS);ratio=statistics.mean(v['us'][j]/v['us'][0] for v in PC_ROWS)
            e=max(v['errors'][j][0] for v in PC_ROWS);r=max(v['errors'][j][1] for v in PC_ROWS)
            lines.append(f'MEAN {name} n={len(PC_ROWS)}/36 us={mean:.2f} vsHH={ratio:.4f} L2={e:.5f} row={r:.5f} PASS')
    for M in (16,32):
        rows=[v for v in PC_SMALL if v['M']==M]
        for j,name in enumerate(('NT1lop','P64s2','P64s3')):
            if rows:lines.append(f'SMALL M{M} {name} n={len(rows)}/36 us={statistics.mean(v["us"][j] for v in rows):.2f} ratio={statistics.mean(v["us"][j]/v["us"][0] for v in rows):.4f} L2={max(v["errors"][j][0] for v in rows):.5f} row={max(v["errors"][j][1] for v in rows):.5f} PASS')
    for label,rows in (('M128',PC_ROWS),('M16_32',PC_SMALL)):
        if rows:
            noise=[x for v in rows for x in v['aa']];lines.append(f'AA {label} delta/abs/max_us={statistics.mean(noise):.3f}/{statistics.mean(abs(v) for v in noise):.3f}/{max(abs(v) for v in noise):.3f}')
    lines.append(f'END M128={len(PC_ROWS)}/36 small={len(PC_SMALL)}/72 lab_s={time.monotonic()-started:.1f} left={deadline-time.monotonic():.1f}')
    return lines
