import statistics
import time

M128_CHOICES=((0,528),(1,528),(2,528),(2,1056),(3,264),(3,528),(3,1056),(4,528),(5,528))
M128_NAMES=('NT1','NT2','masked')+tuple('A'+str(i) for i in range(len(M128_CHOICES)))


def m128_gate(graph,ds,wants):
    err=row=0.
    for data,want in zip(ds,wants):
        graph.refresh(data);got=graph.replay();torch.cuda.synchronize()
        if not bool(torch.isfinite(got).all()):return float('inf'),float('inf')
        delta=got.float()-want.float();den=want.float().norm(dim=-1)
        err=max(err,float(delta.norm()/want.float().norm().clamp_min(1e-12)))
        row=max(row,float((delta.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max()))
    return err,row


def m128_measure(w,native,ctrl,bank,ds,flush):
    wants=dc_oracles(ds,w)
    funcs=[lambda d:rv_kernel_fn(d,dict(native,nt=1)),lambda d:rv_kernel_fn(d,dict(native,nt=2)),lambda d:pp_reference(d,ctrl,'masked')]
    funcs.extend(lambda d,c=c:m128_call(d,bank,*c) for c in M128_CHOICES)
    graphs=[];errors=[]
    for f in funcs:
        graph=MWGraph(f,ds[0]);graphs.append(graph);errors.append(m128_gate(graph,ds,wants))
    if any(e>.01 or r>.05 for e,r in errors[:3]):raise RuntimeError('M128 controls failed FP32 '+str(errors[:3]))
    times=[[] for g in graphs];start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    for i in range(6):
        for g in graphs:g.refresh(ds[i%len(ds)])
        order=list(range(len(graphs)));order=order[i%len(order):]+order[:i%len(order)]
        if i%2:order.reverse()
        for j in order:
            if errors[j][0]>.01 or errors[j][1]>.05:continue
            flush.zero_();start.record();graphs[j].replay();end.record();end.synchronize()
            times[j].append(start.elapsed_time(end)*1000.)
    return [(statistics.median(v) if v else float('inf'),*e) for v,e in zip(times,errors)]


def m128_phases(data,bank,flush):
    counts=torch.bincount(data['topk_ids'][data['topk_ids']>=0],minlength=32);experts=int((counts>0).sum());rows=[]
    for which,grid in ((0,528),(2,528),(3,528),(4,528)):
        fs=m128_work(data,bank,which,grid)
        for f in fs:f()
        ts=[]
        for phase,f in enumerate(fs):
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):f()
            vals=[];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            for i in range(4):
                if phase>=1:fs[0]()
                if phase>=2:fs[1]()
                flush.zero_();a.record();graph.replay();b.record();b.synchronize();vals.append(a.elapsed_time(b)*1000.)
            ts.append(statistics.median(vals))
        rows.append(f'PH cfg={which} g={grid} route,w13_act,w2_final='+','.join(f'{t:.1f}' for t in ts)+f' uniqGBs={experts*18*1048576/ts[1]/1000:.0f},{experts*9*1048576/ts[2]/1000:.0f}')
    return rows


def lab(deadline,weights,state=None):
    dev=next(v.device for v in weights.values() if torch.is_tensor(v));flush=torch.empty(128<<20,device=dev,dtype=torch.uint8)
    lines=['META M128; shared fused.cu producer + expert-interleaved schedule, fixed64/128 and runtime16/32/64/128 token MMA selected OUTSIDE K loop. NT128 reads each touched expert Q+raw E4M3 once (M128); three launches route/reset, W13+SwiGLU, W2+norm+combine. No prepared expanded bank.',
           'META exact R0 + 12 public-router weight seeds, three fresh inputs/captured variant; FP32 finite,L2<=.01,row<=.05. Six rotated cold128MiB replays. GB/s is nominal UNIQUE Q+SF bytes/time, not a DRAM counter; phases include fused epilogues. ORDER '+','.join(M128_NAMES)]
    for j,c in enumerate(CONFIGS):lines.append(f'CFG{j} NT,K,S,SK,adapt={c} res13={module().m128_info(j,1)} res2={module().m128_info(j,0)}')
    lines.append('CHOICES '+','.join(f'A{i}={c[0]}g{c[1]}' for i,c in enumerate(M128_CHOICES)))
    records=[]
    for i in range(12):
        if time.monotonic()>deadline-12:lines.append(f'INCOMPLETE before_seed={i}');break
        w=dc_weights(5353+7919*i,dev);native,ctrl=pp_control_prepare(w);bank=m128_prepare(w)
        if i==0:
            ds=dc_fixtures(w,128,True);vals=m128_measure(w,native,ctrl,bank,ds,flush)
            lines.append('R0 us='+','.join(f'{v[0]:.1f}' for v in vals)+f' gate={max(v[1] for v in vals):.5f}/{max(v[2] for v in vals):.5f}')
            lines.extend(m128_phases(ds[0],bank,flush))
        ds=dc_fixtures(w,128);vals=m128_measure(w,native,ctrl,bank,ds,flush)
        counts=torch.bincount(ds[0]['topk_ids'][ds[0]['topk_ids']>=0],minlength=32);unique=int((counts>0).sum())*28311552
        lines.append(f'S{i} P/E/h={int(counts.sum())}/{int((counts>0).sum())}/{int(counts.max())} us='+','.join(f'{v[0]:.1f}' for v in vals)+f' gate={max(v[1] for v in vals):.5f}/{max(v[2] for v in vals):.5f}')
        records.append((vals,unique))
    for j,name in enumerate(M128_NAMES):
        if records:
            vals=[v[j] for v,u in records];ok=all(e<=.01 and r<=.05 for t,e,r in vals)
            lines.append(f'AGG {name} n={len(vals)} mean={statistics.mean(v[0] for v in vals):.1f} uniqGBs={statistics.mean(u/v[j][0]/1000 for v,u in records):.0f} l2={max(v[1] for v in vals):.5f} row={max(v[2] for v in vals):.4f} '+('PASS' if ok else 'REJECT'))
    lines.append(f'END left={deadline-time.monotonic():.1f}s')
    return lines
