import time
import statistics

WX_CHOICES=tuple((c,g,1) for c in (0,1) for g in (528,1056,2112))
WX_NAMES=('masked','P16','P32')+tuple('CTA'+str(c)+'g'+str(g) for c,g,m in WX_CHOICES)


def wx_measure(w,ctrl,pb,fb,ds,flush):
    funcs=[lambda d:pp_reference(d,ctrl,'masked'),lambda d:pipe_call(d,pb,1,1056),lambda d:pipe_call(d,pb,5,1056)]
    funcs.extend(lambda d,c=c,g=g,m=m:fx_call(d,fb,c,g,m) for c,g,m in WX_CHOICES)
    wx_need(4.,'oracle');wants=dc_oracles(ds,w);gs=[];errs=[]
    for f in funcs:
        wx_need(2.5,'graph/JIT');g=MWGraph(f,ds[0]);e=0.;row=0.
        for d,want in zip(ds,wants):
            wx_need(.5,'gate');g.refresh(d);got=g.replay();torch.cuda.synchronize()
            if not bool(torch.isfinite(got).all()):e=row=float('inf');continue
            diff=got.float()-want.float();den=want.float().norm(dim=-1)
            e=max(e,float(diff.norm()/want.float().norm().clamp_min(1e-12)))
            row=max(row,float((diff.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max()))
        gs.append(g);errs.append((e,row))
    if any(e>.01 or r>.05 for e,r in errs[:3]):raise RuntimeError('W2FIX control gate '+str(errs[:3]))
    ts=[[] for g in gs];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for rep in range(8):
        wx_need(.5,'timing repeat')
        for g in gs:g.refresh(ds[rep%3])
        order=list(range(len(gs)));order=order[rep%len(order):]+order[:rep%len(order)]
        if rep%2:order.reverse()
        for j in order:
            if errs[j][0]>.01 or errs[j][1]>.05:continue
            wx_need(.2,'timing graph');flush.zero_();a.record();gs[j].replay();b.record();b.synchronize();ts[j].append(a.elapsed_time(b)*1000.)
    return [(statistics.median(t) if t else float('inf'),e,r) for t,(e,r) in zip(ts,errs)]


def wx_diagnostics(w,fb,flush):
    wx_need(4.,'R0 oracle');ds=dc_fixtures(w,256,True);data=ds[0];out=[];want=dc_oracles([data],w)[0]
    for which in (0,1):
        nt=FX_CONFIGS[which][0];ids=data['topk_ids'];cnt=torch.bincount(ids[(ids>=0)&(ids<32)],minlength=32)
        groups=int(((cnt+nt-1)//nt).sum());unique=int((cnt>0).sum())
        out.append(f'R0 c={which} P={int(cnt.sum())} E={unique} groups={groups} CTA_QSF=147456 request_MB13,2={groups*128*147456/1e6:.1f},{groups*64*147456/1e6:.1f}')
        for mode in ((0,1) if which==0 else (1,)):
            wx_need(3.,'phase graph/JIT');plan=fx_plan(data,fb,which,1056,mode);fs=plan['steps']
            for f in fs:got=f()
            torch.cuda.synchronize();diff=got.float()-want.float();den=want.float().norm(dim=-1)
            err=float(diff.norm()/want.float().norm().clamp_min(1e-12));row=float((diff.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max())
            if not bool(torch.isfinite(got).all()) or err>.01 or row>.05:raise RuntimeError(f'R0 c={which} m={mode} gate {err} {row}')
            vals=[]
            for phase,f in enumerate(fs):
                wx_need(1.,'phase capture');g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):f()
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);times=[]
                for rep in range(3):
                    if phase>=1:fs[0]()
                    if phase>=2:fs[1]()
                    if phase>=3:fs[2]()
                    wx_need(.3,'phase repeat');flush.zero_();a.record();g.replay();b.record();b.synchronize();times.append(a.elapsed_time(b)*1000.)
                vals.append(statistics.median(times))
            out.append(f'PH c={which} m={mode} us='+','.join(f'{v:.1f}' for v in vals)+f' req_TBs13,2={groups*128*147456/vals[1]/1e6:.2f},{groups*64*147456/vals[2]/1e6:.2f} e={err:.4f}')
            wx_need(1.,'phase clocks');fs[0]();fs[1]();plan['profile'].zero_();fx_profile(plan,fb);torch.cuda.synchronize()
            p=plan['profile'].view(torch.int64).reshape(-1,8)[:groups*64].cpu();last=p[:,6]>0
            med=[int(p[:,j].median()) for j in range(3)]
            med.extend(int(p[last,j].median()) if bool(last.any()) else 0 for j in range(3,6))
            out.append(f'CY c={which} m={mode} main,YSS,group,norm,token,combine='+','.join(map(str,med))+f' combine_max={int(p[:,5].max())}')
    return out


def lab(deadline,weights,state=None):
    global WX_DEADLINE
    WX_DEADLINE=deadline;begin=time.monotonic();records=[];lines=['META M256;12 public weight/router seeds;3 fresh inputs,8 rotated cold128MiB graphs. finite/L2<=.01,row<=.05. CTA=W2 Y+SS then vector token CTA; m0=old,m1=CTA,m2=HOIST. Gates precede timing; partial n never qualifies.']
    lines.append('ORDER '+','.join(WX_NAMES))
    peak=0.;w=None;pb=None;fb=None
    try:
        wx_need(40.,'before native compile');t=time.monotonic();mod=wx_module();lines.append(f'BUILD seconds={time.monotonic()-t:.1f} left={deadline-time.monotonic():.1f}')
        for c in range(2):
            lines.append(f'RES c={c} w13='+str(mod.fx_info(c,1,1))+' w2_CTA='+str(mod.fx_info(c,0,1)))
        dev=next(v.device for v in weights.values() if torch.is_tensor(v));flush=torch.empty(128<<20,device=dev,dtype=torch.uint8)
        for i in range(12):
            wx_need(max(8. if i==0 else 4.,peak*1.5+1.),'before seed'+str(i));t=time.monotonic()
            w=dc_weights(5353+7919*i,dev);ctrl=wx_control_prepare(w);wx_need(3.,'packed producer/JIT');pb=pipe_prepare(w);fb=fx_from_packed(pb)
            ds=dc_fixtures(w,256);vals=wx_measure(w,ctrl,pb,fb,ds,flush);records.append(vals)
            ids=ds[0]['topk_ids'];cnt=torch.bincount(ids[(ids>=0)&(ids<32)],minlength=32)
            lines.append(f'S{i} P={int(cnt.sum())} E={int((cnt>0).sum())} H={int(cnt.max())} us='+','.join(f'{v[0]:.0f}' for v in vals))
            peak=max(peak,time.monotonic()-t)
        wx_need(6.,'R0 diagnostics');w=dc_weights(5353,dev);wx_need(6.,'R0 prepare/JIT');pb=pipe_prepare(w);fb=fx_from_packed(pb);lines.extend(wx_diagnostics(w,fb,flush))
    except WxBudget as e:
        lines.append('BUDGET '+str(e))
    for j,name in enumerate(WX_NAMES):
        vals=[r[j] for r in records]
        if vals:
            good=all(e<=.01 and row<=.05 for _,e,row in vals);ts=[v[0] for v in vals]
            lines.append(f'AGG {name} n={len(ts)} mean={statistics.mean(ts):.1f} med={statistics.median(ts):.1f} l2={max(v[1] for v in vals):.5f} row={max(v[2] for v in vals):.4f} '+('PASS' if good else 'REJECT'))
    lines.append(f'COMPLETE seeds={len(records)}/12 used={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}')
    return lines
