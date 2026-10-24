import statistics
import time

FUSED_CHOICES=((0,264),(0,528),(0,1056),(1,528),(2,528),(2,1056),(3,528),(3,1056),
               (4,528),(5,528),(5,1056),(6,1056),(7,1056))
FUSED_NAMES=('NT1','NT2','masked','VFIX32','P128','P256')+tuple('F'+str(i) for i in range(len(FUSED_CHOICES)))


def fused_fixtures(w,balanced=False,r0=False):
    ds=dc_fixtures(w,256,r0)
    if balanced:
        for d in ds:
            ids=torch.full((256,8),255,device=w['w13'].device,dtype=torch.int32)
            ids[:,0]=torch.arange(256,device=ids.device)%32;d['topk_ids']=ids
    return ds


def fused_measure(w,old,ctrl,vb,pb,fb,ds,flush):
    funcs=[lambda d:rv_kernel_fn(d,dict(old,nt=1)),lambda d:rv_kernel_fn(d,dict(old,nt=2)),
        lambda d:pp_reference(d,ctrl,'masked'),lambda d:vec_call(d,vb,32,528),
        lambda d:pipe_call(d,pb,1,528),lambda d:pipe_call(d,pb,4,528)]
    funcs.extend(lambda d,c=c:ft_call(d,fb,*c) for c in FUSED_CHOICES)
    gs=[MWGraph(f,ds[0]) for f in funcs];wants=dc_oracles(ds,w);errors=[]
    for g in gs:
        err,row=0.,0.
        for data,want in zip(ds,wants):
            g.refresh(data);got=g.replay();torch.cuda.synchronize()
            if not bool(torch.isfinite(got).all()):err=float('inf');row=float('inf');continue
            diff=got.float()-want.float();den=want.float().norm(dim=-1)
            err=max(err,float(diff.norm()/want.float().norm().clamp_min(1e-12)))
            row=max(row,float((diff.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max()))
        errors.append((err,row))
    if any(e>.01 or r>.05 for e,r in errors[:3]):raise RuntimeError('M256 control gate '+str(errors[:3]))
    nts=(8,16,0,32,16,16)+tuple(FT_CONFIGS[c[0]][0] for c in FUSED_CHOICES)
    req=[]
    for data in ds:
        ids=data['topk_ids'];cnt=torch.bincount(ids[(ids>=0)&(ids<32)],minlength=32).cpu().tolist()
        req.append([sum((n+nt-1)//nt for n in cnt)*28311552 if nt else 0 for nt in nts])
    times=[[] for _ in gs];rates=[[] for _ in gs];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for i in range(8):
        for g in gs:g.refresh(ds[i%3])
        order=list(range(len(gs)));order=order[i%len(order):]+order[:i%len(order)]
        if i%2:order.reverse()
        for j in order:
            if errors[j][0]>.01 or errors[j][1]>.05:continue
            flush.zero_();a.record();gs[j].replay();b.record();b.synchronize();us=a.elapsed_time(b)*1000.
            times[j].append(us);rates[j].append(req[i%3][j]/us/1e3)
    return [(statistics.median(t) if t else float('inf'),e,r,statistics.median(bw) if bw else 0.) for t,(e,r),bw in zip(times,errors,rates)]


def fused_phases(data,fb,flush):
    rows=[]
    for which,grid in ((0,528),(2,528),(3,528),(5,1056),(7,1056)):
        fs=ft_work(data,fb,which,grid)
        for f in fs:f()
        vals=[]
        for phase,f in enumerate(fs):
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):f()
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);ts=[]
            for _ in range(6):
                # Completion tickets belong to one invocation. Reset before
                # timing an isolated GEMM; regenerate W13's input for W2.
                if phase>=1:fs[0]()
                if phase>=2:fs[1]()
                flush.zero_();a.record();graph.replay();b.record();b.synchronize();ts.append(a.elapsed_time(b)*1000.)
            vals.append(statistics.median(ts))
        rows.append(f'PH {which}/{grid} route,w13,w2='+','.join(f'{t:.1f}' for t in vals))
    return rows


def lab(deadline,weights,state=None):
    dev=next(v.device for v in weights.values() if torch.is_tensor(v));flush=torch.empty(128<<20,device=dev,dtype=torch.uint8)
    lines=['META M256;12 public-router seeds,3 fresh inputs,8 cold128MiB graphs. R0 exact IDs; BAL32 synthetic. VFIX aligned; fused route/W13+SwiGLU/W2+norm+combine. Raw NVFP4/E4M3;FP32 GS. Gates finite,L2<=.01,row<=.05. us rounded; QSF_GBs nominal requests,not DRAM.']
    lines.append('ORDER '+','.join(FUSED_NAMES))
    resource=[]
    for i,c in enumerate(FT_CONFIGS):
        a=ft_module().ft_info(i,1);b=ft_module().ft_info(i,0)
        resource.append('/'.join(map(str,c))+':'+','.join(map(str,(a[0],b[0],a[1],b[1],a[3],a[4],b[4]))))
    lines.append('CFG NT/K/S/SK:r13,r2,local13,local2,dyn,res13,res2 '+';'.join(resource))
    records=[]
    for i in range(12):
        if time.monotonic()>deadline-15:lines.append(f'INCOMPLETE before_seed={i}');break
        w=dc_weights(5353+7919*i,dev);old,ctrl=pp_control_prepare(w)
        vb=vec_prepare(w);pb=dict(vb,mod=pipe_module());fb=ft_from_packed(vb)
        if i==0:
            for label,ds in (('R0',fused_fixtures(w,r0=True)),('BAL32',fused_fixtures(w,balanced=True))):
                vals=fused_measure(w,old,ctrl,vb,pb,fb,ds,flush)
                counts=torch.bincount(ds[0]['topk_ids'][(ds[0]['topk_ids']>=0)&(ds[0]['topk_ids']<32)],minlength=32)
                lines.append(f'{label} P={int(counts.sum())} E={int((counts>0).sum())} H={int(counts.max())} us='+','.join(f'{v[0]:.0f}' for v in vals)+f' l2={max(v[1] for v in vals):.5f} row={max(v[2] for v in vals):.4f}')
                lines.extend(label+' '+s for s in fused_phases(ds[0],fb,flush))
        ds=fused_fixtures(w);vals=fused_measure(w,old,ctrl,vb,pb,fb,ds,flush)
        counts=torch.bincount(ds[0]['topk_ids'][ds[0]['topk_ids']>=0],minlength=32)
        lines.append(f'S{i} P={int(counts.sum())} E={int((counts>0).sum())} H={int(counts.max())} us='+','.join(f'{v[0]:.0f}' for v in vals))
        records.append(vals)
    for j,name in enumerate(FUSED_NAMES):
        vals=[r[j] for r in records]
        if vals:
            good=all(e<=.01 and r<=.05 for _,e,r,_ in vals);ts=[v[0] for v in vals]
            bw='NA' if name=='masked' else f'{statistics.mean(v[3] for v in vals):.1f}'
            lines.append(f'AGG {name} n={len(ts)} mean={statistics.mean(ts):.1f} med={statistics.median(ts):.1f} QSF_GBs={bw} l2={max(v[1] for v in vals):.5f} row={max(v[2] for v in vals):.4f} '+('PASS' if good else 'REJECT'))
    return lines
