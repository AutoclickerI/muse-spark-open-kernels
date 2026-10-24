import time
import statistics

WI_NAMES=('CTA1g2112','masked','typed_LOP','vector_LOP','vector_LOP_recW13','C1g2112','vector_LOP_fullrec','mixed_seq','mixed_parallel','quad_raw','quad_fullrec','mixed_parallel_quad')
WI_POLICIES=(-1,0,1,2,3,4,5,6,7,8,9)


def issue_measure(w,bank,ctrl,ds,flush):
    funcs=[lambda d:wi_call(d,bank,-1),lambda d:pp_reference(d,ctrl,'masked')]
    funcs.extend(lambda d,p=p:wi_call(d,bank,p) for p in (0,1,2,3,4,5,6,7,8,9));wx_need(4.,'FP32 oracle');wants=dc_oracles(ds,w);graphs=[];errors=[]
    for f in funcs:
        wx_need(3.,'graph/JIT');g=MWGraph(f,ds[0]);e=0.;r=0.
        for d,want in zip(ds,wants):
            wx_need(.5,'fresh gate');g.refresh(d);got=g.replay();torch.cuda.synchronize()
            if not bool(torch.isfinite(got).all()):e=r=float('inf');continue
            diff=got.float()-want.float();den=want.float().norm(dim=-1);e=max(e,float(diff.norm()/want.float().norm().clamp_min(1e-12)));r=max(r,float((diff.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max()))
        graphs.append(g);errors.append((e,r))
    if any(e>.01 or r>.05 for e,r in errors[:2]):raise RuntimeError('W13 control gate '+str(errors[:2]))
    ts=[[] for g in graphs];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for rep in range(8):
        wx_need(.5,'timing repeat')
        for g in graphs:g.refresh(ds[rep%3])
        order=list(range(len(graphs)));order=order[rep%len(order):]+order[:rep%len(order)]
        if rep%2:order.reverse()
        for j in order:
            if errors[j][0]>.01 or errors[j][1]>.05:continue
            wx_need(.2,'graph replay');flush.zero_();a.record();graphs[j].replay();b.record();b.synchronize();ts[j].append(a.elapsed_time(b)*1000.)
    return [(statistics.median(t) if t else float('inf'),e,r) for t,(e,r) in zip(ts,errors)]


def issue_phases(w,bank,flush):
    wx_need(4.,'R0 oracle');d=dc_fixtures(w,256,True)[0];want=dc_oracles([d],w)[0];ids=d['topk_ids'];cnt=torch.bincount(ids[(ids>=0)&(ids<32)],minlength=32);groups=int(((cnt+31)//32).sum());unique=int((cnt>0).sum());lines=[f'R0 P={int(cnt.sum())} E={unique} hot={int(cnt.max())} groups32={groups} fill={int(cnt.sum())/(groups*32):.3f} qsf_reqMB={groups*128*147456/1e6:.1f} uniqueMB={unique*8192*4096*9/16/1e6:.1f}']
    for policy in (-1,1,3,4,6,7,8,9):
        wx_need(3.,'phase capture');plan=wi_plan(d,bank,policy);steps=plan['steps']
        for f in steps:got=f()
        torch.cuda.synchronize();diff=got.float()-want.float();e=float(diff.norm()/want.float().norm());den=want.float().norm(dim=-1);row=float((diff.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max())
        if not bool(torch.isfinite(got).all()) or e>.01 or row>.05:raise RuntimeError(f'R0 policy={policy} gate {e} {row}')
        vals=[]
        for phase,f in enumerate(steps):
            wx_need(1.,'phase graph');g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):f()
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);ts=[]
            for rep in range(4):
                for pre in steps[:phase]:pre()
                wx_need(.2,'phase repeat');flush.zero_();a.record();g.replay();b.record();b.synchronize();ts.append(a.elapsed_time(b)*1000.)
            vals.append(statistics.median(ts))
        lines.append(f'PH p={policy} route,w13,w2,finish='+','.join(f'{v:.1f}' for v in vals)+f' W13reqTBs={groups*128*147456/vals[1]/1e6:.2f} uniqueTBs={unique*8192*4096*9/16/vals[1]/1e6:.2f} e={e:.5f}')
    return lines


def lab(deadline,weights,state=None):
    global WX_DEADLINE
    WX_DEADLINE=deadline;start=time.monotonic();records=[];peak=0.
    lines=['META M256 W13-only edits; p3-6/8/9 use same recoded W2,others raw W2. K128/S4/grid2112 fixed; p5/6 split <=16-row experts N16,others N32; seq/parallel. Typed shared align1024; Q16B/SF16B vector copies; immediate LOP3; p7+ quad scale broadcasts; recW13 uses exact scale recode. 12same public weight/router seeds x3fresh inputs x8rotated cold128MiB graphs. Gates finite/L2<=.01,row<=.05; partial n not a win.']
    lines.append('ORDER '+','.join(WI_NAMES))
    try:
        wx_need(40.,'before native compile');t=time.monotonic();mod=fx_module();lines.append(f'BUILD s={time.monotonic()-t:.1f} left={deadline-time.monotonic():.1f}')
        lines.append('RECBASE W13/W2='+str(mod.rc_info(1,1,1))+str(mod.rc_info(1,0,1)))
        lines.append('BASE W13regs,local,static,dynamic,res='+str(mod.fx_info(1,1,1)))
        lines.append('MIXRES cold/hot='+str(mod.wi_mixed_info(0))+str(mod.wi_mixed_info(1)))
        for p in (0,1,2):lines.append(f'RES p={p} regs,local,static,dynamic,res='+str(mod.wi_info(p)))
        dev=next(v.device for v in weights.values() if torch.is_tensor(v));flush=torch.empty(128<<20,device=dev,dtype=torch.uint8)
        for i in range(12):
            wx_need(max(8. if i==0 else 4.,peak*1.5+1.),'before seed'+str(i));t=time.monotonic();w=dc_weights(5353+7919*i,dev);ctrl=wx_control_prepare(w);wx_need(4.,'bank repack/JIT');bank=wi_prepare(w);ds=dc_fixtures(w,256);vals=issue_measure(w,bank,ctrl,ds,flush);records.append(vals)
            ids=ds[0]['topk_ids'];cnt=torch.bincount(ids[(ids>=0)&(ids<32)],minlength=32);lines.append(f'S{i} P={int(cnt.sum())} E={int((cnt>0).sum())} H={int(cnt.max())} us='+','.join(f'{v[0]:.0f}' for v in vals));peak=max(peak,time.monotonic()-t)
        wx_need(6.,'R0 diagnostics');w=dc_weights(5353,dev);wx_need(4.,'R0 repack/JIT');bank=wi_prepare(w);lines.extend(issue_phases(w,bank,flush))
    except WxBudget as e:lines.append('BUDGET '+str(e))
    for j,name in enumerate(WI_NAMES):
        vals=[r[j] for r in records]
        if vals:
            ts=[v[0] for v in vals];good=all(e<=.01 and r<=.05 for _,e,r in vals)
            lines.append(f'AGG {name} n={len(vals)} mean={statistics.mean(ts):.1f} med={statistics.median(ts):.1f} /base={statistics.mean(r[j][0]/r[0][0] for r in records):.3f} l2={max(v[1] for v in vals):.5f} row={max(v[2] for v in vals):.4f} '+('PASS' if good else 'REJECT'))
    lines.append(f'COMPLETE seeds={len(records)}/12 used={time.monotonic()-start:.1f} left={deadline-time.monotonic():.1f}');return lines
