MX_SEEDS=12
MX_ROWS=[]
MX_PHASES=[]


def mx_recipes():
    out=[('C1g2112',(0,32),(0,32),0,2112,True)]
    for a in MX_FIRST:
        for b in MX_SECOND:out.append((f'A{a[0]}n{a[1]}_B{b[0]}n{b[1]}',a,b,MX_DECODE,2112,False))
    out.append(('H16u8g1056',(16,32),(16,32),0,1056,False))
    return tuple(out)


MX_RECIPES=mx_recipes()


def mx_gate(g,want,label):
    out=g.replay();torch.cuda.synchronize();e,r=mx_error(out,want)
    if e>.01 or r>.05:raise RuntimeError(f'FP32 {label}:L2={e},row={r}')
    return e,r


def mx_pair(a,b,flush,event0,event1,repeats=6):
    aa=[];bb=[];delta=[]
    for repeat in range(repeats):
        mx_need(.3,'paired cold order')
        order=((a,aa),(b,bb)) if repeat%2==0 else ((b,bb),(a,aa))
        pair=[]
        for graph,samples in order:
            flush.zero_();event0.record();graph.replay();event1.record();event1.synchronize()
            us=event0.elapsed_time(event1)*1000.;samples.append(us);pair.append(us)
        if a is b:delta.append(pair[1]-pair[0])
    return statistics.median(aa),statistics.median(bb),delta


def mx_measure(w,bank,ds,flush):
    mx_need(4.,'FP32 oracles');wants=dc_oracles(ds,w);graphs=[]
    for name,a,b,decode,grid,control in MX_RECIPES:
        mx_need(2.,'graph/JIT '+name)
        graphs.append(MWGraph(lambda d,a=a,b=b,q=decode,g=grid,c=control:mx_call(d,bank,a,b,q,g,c),ds[0]))
    event0=torch.cuda.Event(enable_timing=True);event1=torch.cuda.Event(enable_timing=True);rows=[]
    for fi,(data,want) in enumerate(zip(ds,wants)):
        errors=[]
        for j,graph in enumerate(graphs):
            mx_need(.5,'fresh fixture gate');graph.refresh(data);errors.append(mx_gate(graph,want,MX_RECIPES[j][0]))
        # All arms consume these actual supplied IDs/scores. A router's FP32
        # cutoff flip is accepted as a new valid fixture, never ID-equality gated.
        base=graphs[0];_,_,noise=mx_pair(base,base,flush,event0,event1)
        values=[0.];ratios=[1.];baselines=[]
        order=list(range(1,len(graphs)));shift=fi%len(order);order=order[shift:]+order[:shift]
        values.extend(0. for _ in order);ratios.extend(0. for _ in order)
        for j in order:
            au,bu,_=mx_pair(base,graphs[j],flush,event0,event1);baselines.append(au);values[j]=bu;ratios[j]=bu/au
        values[0]=statistics.median(baselines)
        for j,graph in enumerate(graphs):
            mx_need(.3,'post-timing gate');e,r=mx_gate(graph,want,'post '+MX_RECIPES[j][0]);errors[j]=(max(e,errors[j][0]),max(r,errors[j][1]))
        ids=data['topk_ids'];cnt=torch.bincount(ids[(ids>=0)&(ids<32)],minlength=32)
        rows.append(dict(fixture=fi,pairs=int(cnt.sum()),experts=int((cnt>0).sum()),hot=int(cnt.max()),us=values,ratios=ratios,errors=errors,aa=noise))
    return rows


def mx_phase_measure(data,bank,a,b,decode,flush,control=False):
    plan=mx_plan(data,bank,a,b,decode,2112,control);steps=plan['components'];vals=[]
    for fn in steps:out=fn()
    for phase,fn in enumerate(steps):
        mx_need(1.,'phase graph');graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):fn()
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);samples=[]
        for repeat in range(3):
            mx_need(.3,'phase repeat')
            for prep in steps[:phase]:prep()
            flush.zero_();start.record();graph.replay();end.record();end.synchronize();samples.append(start.elapsed_time(end)*1000.)
        vals.append(statistics.median(samples))
    return vals


def lab(deadline,weights,state=None):
    global MX_DEADLINE,MX_ROWS,MX_PHASES
    MX_DEADLINE=deadline;MX_ROWS=[];MX_PHASES=[];beg=time.monotonic();peak=0.;lines=[
        f'META M256 packed-u8 expert mix; decode={MX_DECODE},threshold8/16/32,N32/64 independently W13/W2;42crosses+C1g2112+H16u8g1056. No widened scale bank. Actual supplied routes:oracle follows FP32 topk flips.',
        'ORDER A rows='+str(MX_FIRST)+' B cols='+str(MX_SECOND)+';12public weight/router seeds x3fixtures,6cold128MiB AB/BA orders per A:B,A:A,FP32 pre/post. Incomplete/failed cannot promote.']
    try:
        mx_need(40.,'before compile');mod=mx_module();lines.append(f'BUILD s={time.monotonic()-beg:.1f} left={deadline-time.monotonic():.1f}')
        lines.append('RES N32/64 W13/W2='+str([mod.rc_info(n,f,1) for n in (1,2) for f in (1,0)]))
        dev=next(v.device for v in weights.values() if torch.is_tensor(v));flush=torch.empty(128<<20,device=dev,dtype=torch.uint8)
        for seed in range(MX_SEEDS):
            mx_need(max(10.,peak*1.3+1.),'before seed'+str(seed));tick=time.monotonic();w=dc_weights(5353+7919*seed,dev)
            mx_need(4.,'bank repack/JIT');bank=mx_prepare(w);ds=dc_fixtures(w,256);rows=mx_measure(w,bank,ds,flush)
            # Fixed phase matrix on fixture0 of each seed: baseline and six
            # coupled expert choices. Components distinguish small/heavy work.
            phases=[]
            for a in ((0,32),)+MX_FIRST:
                vals=mx_phase_measure(ds[0],bank,a,a,MX_DECODE,flush,control=a[0]==0);phases.append(dict(name=str(a),us=vals))
            MX_ROWS.extend(dict(seed=seed,**row) for row in rows);MX_PHASES.extend(dict(seed=seed,**p) for p in phases)
            baseline=statistics.mean(r['us'][0] for r in rows);lines.append(f'S{seed} P/E/H={rows[0]["pairs"]}/{rows[0]["experts"]}/{rows[0]["hot"]} C1={baseline:.1f} gates={len(rows)*len(MX_RECIPES)}PASS');peak=max(peak,time.monotonic()-tick)
    except MXBudget as e:lines.append('BUDGET '+str(e))
    if MX_ROWS:
        mean=[statistics.mean(r['us'][j] for r in MX_ROWS) for j in range(len(MX_RECIPES))]
        median=[statistics.median(r['us'][j] for r in MX_ROWS) for j in range(len(MX_RECIPES))]
        ratios=[statistics.mean(r['ratios'][j] for r in MX_ROWS) for j in range(len(MX_RECIPES))]
        errors=[(max(r['errors'][j][0] for r in MX_ROWS),max(r['errors'][j][1] for r in MX_ROWS)) for j in range(len(MX_RECIPES))]
        lines.append(f'C1 n={len(MX_ROWS)}/{MX_SEEDS*3} mean/med={mean[0]:.2f}/{median[0]:.2f} H16g1056={mean[-1]:.2f}/{median[-1]:.2f} ratio={ratios[-1]:.4f}')
        for ai,a in enumerate(MX_FIRST):
            js=range(1+ai*len(MX_SECOND),1+(ai+1)*len(MX_SECOND))
            lines.append(f'MEAN {a} '+','.join(f'{mean[j]:.1f}' for j in js))
            lines.append(f'PAIR {a} '+','.join(f'{ratios[j]:.4f}' for j in js))
            lines.append(f'MED {a} '+','.join(f'{median[j]:.1f}' for j in js))
        e=max(v[0] for v in errors);r=max(v[1] for v in errors);lines.append(f'GATES fixed={len(MX_RECIPES)} fixtures={len(MX_ROWS)} maxL2/row={e:.6f}/{r:.6f} PASS')
        for a in ((0,32),)+MX_FIRST:
            ph=[p['us'] for p in MX_PHASES if p['name']==str(a)]
            if ph:lines.append(f'PH {a} n={len(ph)} route/As/Ah/Bs/Bh/f='+','.join(f'{statistics.mean(v[j] for v in ph):.1f}' for j in range(6)))
        noise=[v for row in MX_ROWS for v in row['aa']];lines.append(f'AA signed/abs/max={statistics.mean(noise):.3f}/{statistics.mean(abs(v) for v in noise):.3f}/{max(abs(v) for v in noise):.3f}')
    lines.append(f'END seeds={len(MX_ROWS)//3}/{MX_SEEDS} used={time.monotonic()-beg:.1f} left={deadline-time.monotonic():.1f}');return lines
