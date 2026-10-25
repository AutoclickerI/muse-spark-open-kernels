RI_NAMES=('S6_grid128','dead_grid128','live_grid1')


def ri_gate(g,want):
    out=g.replay();torch.cuda.synchronize();return ri_error(out,want)


def ri_pair(a,b,flush,ev0,ev1):
    aa=[];bb=[]
    for repeat in range(6):
        ri_need(.3,'cold pair')
        order=((a,aa),(b,bb)) if repeat%2==0 else ((b,bb),(a,aa))
        for graph,samples in order:
            flush.zero_();ev0.record();graph.replay();ev1.record();ev1.synchronize();samples.append(ev0.elapsed_time(ev1)*1000.)
    return statistics.median(aa),statistics.median(bb),[b-a for a,b in zip(aa,bb)]


def ri_measure(w,bank,ds,flush):
    ri_need(4.,'FP32 oracle');wants=dc_oracles(ds,w);graphs=[]
    for stage in range(3):
        ri_need(2.,'graph/JIT');graphs.append(MWGraph(lambda d,s=stage:ri_call(d,bank,s),ds[0]))
    rows=[];ev0=torch.cuda.Event(enable_timing=True);ev1=torch.cuda.Event(enable_timing=True)
    for fi,(data,want) in enumerate(zip(ds,wants)):
        errors=[]
        for g in graphs:g.refresh(data);errors.append(ri_gate(g,want))
        bit=[];base=graphs[0].replay().clone()
        for g in graphs:bit.append(bool(torch.equal(base,g.replay())))
        au,_,noise=ri_pair(graphs[0],graphs[0],flush,ev0,ev1);us=[au];ratio=[1.]
        for stage in (1,2):
            a,b,_=ri_pair(graphs[0],graphs[stage],flush,ev0,ev1);us.append(b);ratio.append(b/a)
        for j,g in enumerate(graphs):
            e,r=ri_gate(g,want);errors[j]=(max(e,errors[j][0]),max(r,errors[j][1]))
        rows.append(dict(fixture=fi,us=us,ratio=ratio,errors=errors,bit=bit,aa=noise))
    return rows


def ri_phases(data,bank,stage,flush):
    plan=ri_plan(data,bank,stage);steps=plan['steps'];vals=[]
    for f in steps:out=f()
    for k,fn in enumerate(steps):
        ri_need(1.,'phase graph');g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):fn()
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);ss=[]
        for _ in range(4):
            ri_need(.3,'phase timing')
            for pre in steps[:k]:pre()
            flush.zero_();a.record();g.replay();b.record();b.synchronize();ss.append(a.elapsed_time(b)*1000.)
        vals.append(statistics.median(ss))
    return vals


def lab(deadline,weights,state=None):
    global RI_DEADLINE,RI_ROWS
    RI_DEADLINE=deadline;RI_ROWS=[];phase=[];start=time.monotonic();peak=0.;lines=[
        'META exact S6 mixed_parallel:0 original route128CTAs,1 delete OUT/state stores same128,2 same live kernel1CTA. GEMM/finalizer/other modes unchanged.12seeds x3fresh routes,cold128MiB AB/BA+A:A.',
        'GATE FP32 MoE TASK global<=.015,row<=.075,finite; stricter .01/.05 and bit equality diagnostic; record failures,time all arms. No router-ID equality gate:oracle follows supplied routes. Stage means are diagnostic; promotion needs identical-input six-load score.']
    try:
        ri_need(40.,'before compile');mod=fx_module();lines.append(f'BUILD s={time.monotonic()-start:.1f} left={deadline-time.monotonic():.1f}')
        dev=next(t.device for t in weights.values() if torch.is_tensor(t));flush=torch.empty(128<<20,device=dev,dtype=torch.uint8)
        for seed in range(12):
            ri_need(max(8.,peak*1.3+1.),'before seed'+str(seed));tick=time.monotonic();w=dc_weights(5353+7919*seed,dev)
            ri_need(4.,'repack/JIT');bank=wi_prepare(w);ds=dc_fixtures(w,256);rows=ri_measure(w,bank,ds,flush)
            ph=[ri_phases(ds[0],bank,s,flush) for s in range(3)]
            RI_ROWS.extend(dict(seed=seed,**r) for r in rows);phase.append(ph)
            lines.append(f'S{seed} us='+','.join(f'{statistics.mean(r["us"][s] for r in rows):.1f}' for s in range(3)));peak=max(peak,time.monotonic()-tick)
    except RIBudget as e:lines.append('BUDGET '+str(e))
    if RI_ROWS:
        for stage,name in enumerate(RI_NAMES):
            e=max(r['errors'][stage][0] for r in RI_ROWS);row=max(r['errors'][stage][1] for r in RI_ROWS);good=e<=.015 and row<=.075
            lines.append(f'MEAN {name} n={len(RI_ROWS)}/36 us={statistics.mean(r["us"][stage] for r in RI_ROWS):.3f} med={statistics.median(r["us"][stage] for r in RI_ROWS):.3f} pair={statistics.mean(r["ratio"][stage] for r in RI_ROWS):.5f} L2/row={e:.6f}/{row:.6f} TASK='+('PASS' if good else 'FAIL')+f' strict_diag={sum(r["errors"][stage][0]>.01 or r["errors"][stage][1]>.05 for r in RI_ROWS)} bit_diag_miss={sum(not r["bit"][stage] for r in RI_ROWS)}')
            if phase:lines.append(f'PH {name} seeds={len(phase)} route/W13/W2/finish='+','.join(f'{statistics.mean(p[stage][k] for p in phase):.2f}' for k in range(4)))
        noise=[a for r in RI_ROWS for a in r['aa']];lines.append(f'AA signed/abs/max={statistics.mean(noise):.3f}/{statistics.mean(abs(a) for a in noise):.3f}/{max(abs(a) for a in noise):.3f}')
    lines.append(f'END seeds={len(RI_ROWS)//3}/12 used={time.monotonic()-start:.1f} left={deadline-time.monotonic():.1f}');return lines
