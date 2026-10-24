import time
import statistics

RC_CHOICES=tuple((c,g) for c in (0,1) for g in (528,1056,2112))
RC_NAMES=('masked','NT1c')+tuple(('C' if enc else 'R')+str(c)+'g'+str(g) for enc in (False,True) for c,g in RC_CHOICES)


def re_weights(seed,dev):
    w=rg_generate_weights(seed,device=dev);w['post_norm_w']=w['post_expert_norm'];w['w13_gs_gate']=w['w13_gs'];w['w13_gs_up']=w['w13_gs'];return w


def re_measure(w,ctrl,st,banks,ds,flush):
    funcs=[lambda d:pp_reference(d,ctrl,'masked'),lambda d:n1_call(d,st)]
    funcs.extend(lambda d,bank=bank,c=c,g=g:rc_call(d,bank,c,g) for bank in banks for c,g in RC_CHOICES)
    wx_need(4.,'FP32 oracle');wants=dc_oracles(ds,w);graphs=[];errors=[]
    for f in funcs:
        wx_need(3.,'graph/JIT compile');graph=MWGraph(f,ds[0]);err=0.;row=0.
        for d,want in zip(ds,wants):
            wx_need(.5,'fresh gate');graph.refresh(d);got=graph.replay();torch.cuda.synchronize()
            if not bool(torch.isfinite(got).all()):err=row=float('inf');continue
            diff=got.float()-want.float();den=want.float().norm(dim=-1);err=max(err,float(diff.norm()/want.float().norm().clamp_min(1e-12)));row=max(row,float((diff.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max()))
        graphs.append(graph);errors.append((err,row))
    if any(e>.01 or r>.05 for e,r in errors[:2]):raise RuntimeError('recoded lab control gate '+str(errors[:2]))
    ts=[[] for _ in graphs];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for rep in range(8):
        wx_need(.5,'timing repeat')
        for g in graphs:g.refresh(ds[rep%3])
        order=list(range(len(graphs)));order=order[rep%len(order):]+order[:rep%len(order)]
        if rep%2:order.reverse()
        for j in order:
            if errors[j][0]>.01 or errors[j][1]>.05:continue
            wx_need(.2,'timing graph');flush.zero_();a.record();graphs[j].replay();b.record();b.synchronize();ts[j].append(a.elapsed_time(b)*1000.)
    return [(statistics.median(t) if t else float('inf'),e,r) for t,(e,r) in zip(ts,errors)]


def re_phases(w,banks,flush):
    wx_need(4.,'R0 oracle');d=dc_fixtures(w,256,True)[0];want=dc_oracles([d],w)[0];lines=[]
    for c in (0,1):
        for bank in banks:
            wx_need(3.,'R0 capture');plan=rc_plan(d,bank,c,1056);steps=plan['steps']
            for f in steps:got=f()
            torch.cuda.synchronize();diff=got.float()-want.float();err=float(diff.norm()/want.float().norm().clamp_min(1e-12));den=want.float().norm(dim=-1);row=float((diff.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max())
            if not bool(torch.isfinite(got).all()) or err>.01 or row>.05:raise RuntimeError(f'R0 c={c} encoded={bank["encoded"]} gate {err} {row}')
            vals=[]
            for phase,f in enumerate(steps):
                wx_need(1.,'R0 phase');g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):f()
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);ts=[]
                for rep in range(3):
                    for pre in steps[:phase]:pre()
                    wx_need(.2,'R0 repeat');flush.zero_();a.record();g.replay();b.record();b.synchronize();ts.append(a.elapsed_time(b)*1000.)
                vals.append(statistics.median(ts))
            lines.append(f'PH c={c} enc={int(bank["encoded"])} route,w13,w2,finish='+','.join(f'{v:.1f}' for v in vals)+f' e={err:.5f}')
    return lines


def lab(deadline,weights,state=None):
    global WX_DEADLINE
    WX_DEADLINE=deadline;begin=time.monotonic();records=[];peak=0.
    lines=['META M256;12 public weight/router seeds,shared W13 GS model contract. 3fresh inputs/8rotated cold128MiB graphs. R=raw,C=lossless exponent/mantissa+FP4 sign/zero fold; timed PRMT conversion. Same four launches/shape per R:C pair. finite/L2<=.01,row<=.05. NT1c=stock recoded NT1. Partial n never qualifies.']
    lines.append('ORDER '+','.join(RC_NAMES))
    try:
        wx_need(40.,'before native compile');t=time.monotonic();mod=rc_module();lines.append(f'BUILD s={time.monotonic()-t:.1f} left={deadline-time.monotonic():.1f}')
        for c in range(2):
            lines.append(f'RES c={c} raw13,2={mod.rc_info(c,1,0)},{mod.rc_info(c,0,0)} rec13,2={mod.rc_info(c,1,1)},{mod.rc_info(c,0,1)}')
        dev=next(v.device for v in weights.values() if torch.is_tensor(v));flush=torch.empty(128<<20,device=dev,dtype=torch.uint8)
        for i in range(12):
            wx_need(max(10. if i==0 else 4.,peak*1.5+1.),'before seed'+str(i));t=time.monotonic();w=re_weights(5353+7919*i,dev)
            ctrl=wx_control_prepare(w);wx_need(4.,'bank repack/JIT compile');banks=rc_prepare(w);wx_need(4.,'stock NT1 pack');st=n1_prepare(w);ds=dc_fixtures(w,256)
            vals=re_measure(w,ctrl,st,banks,ds,flush);records.append(vals);ids=ds[0]['topk_ids'];cnt=torch.bincount(ids[(ids>=0)&(ids<32)],minlength=32)
            lines.append(f'S{i} P={int(cnt.sum())} E={int((cnt>0).sum())} H={int(cnt.max())} us='+','.join(f'{v[0]:.0f}' for v in vals));peak=max(peak,time.monotonic()-t)
        wx_need(6.,'R0 diagnostics');w=re_weights(5353,dev);wx_need(4.,'R0 bank repack/JIT');banks=rc_prepare(w);lines.extend(re_phases(w,banks,flush))
    except WxBudget as e:lines.append('BUDGET '+str(e))
    for j,name in enumerate(RC_NAMES):
        vals=[r[j] for r in records]
        if vals:
            good=all(e<=.01 and row<=.05 for _,e,row in vals);ts=[v[0] for v in vals];paired=''
            if j>=8:paired=f' C:R={statistics.mean(r[j][0]/r[j-6][0] for r in records):.3f}'
            lines.append(f'AGG {name} n={len(vals)} mean={statistics.mean(ts):.1f} med={statistics.median(ts):.1f}{paired} l2={max(v[1] for v in vals):.5f} row={max(v[2] for v in vals):.4f} '+('PASS' if good else 'REJECT'))
    lines.append(f'COMPLETE seeds={len(records)}/12 used={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}');return lines
