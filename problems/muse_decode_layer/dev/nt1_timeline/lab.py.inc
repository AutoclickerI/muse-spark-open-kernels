class TimelineGraph:
    def __init__(self,fn):
        fn();torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.out=fn()
    def replay(self):self.graph.replay();return self.out


def timeline_timing(old,new,repeats=12):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    aa,bb,ratios=[],[],[]
    def measured(graph):
        lab_cold_l2();a.record();graph.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
    for rep in range(repeats):
        order=(old,new) if rep%2==0 else (new,old);v=[measured(g) for g in order]
        x,y=v if rep%2==0 else v[::-1];aa.append(x);bb.append(y);ratios.append(y/x)
    return statistics.median(aa),statistics.median(bb),statistics.median(ratios)


def task_check(out,want):
    """TASK.md: output relative L2 <= 1.5% vs FP32, every row <= 5x (near-zero rows vs a tenth of the mean row), finite."""
    err,row=nt1_error(out,want);ok=bool(torch.isfinite(out).all()) and err<=.015 and row<=.075
    return ok,err,row


DIAG_BY_M={32:('RTRACEZZP','RTRACEWPUB')}  # v10: critical paths + tile/publish split
# R = the decode layer's context (router node + MoE node). R95/R93 = 849e4017's winners; R77 drops NREG, R69 drops
# NREG and RSQ; ECAP = INC on the same routes with every local expert capped at 8 tokens (DIAGNOSTIC, no gate): the
# most that decode-once / no-reread for multi-block experts could save.
BASE_FLAGS={1:95,8:69,16:69,32:69}
SMALLM_ARMS={1:dict(E={},R={'RBASE':95,'R65615':65615})}
for _m in (8,16,32):
    SMALLM_ARMS[_m]=dict(E={},R={'RBASE':69,'R65605':65605,'R73797':73797,'R139333':139333,'R204869':204869})


def cap_fixture(d,cap=8):
    """Same fixture, each local expert keeps its first `cap` (token, slot) pairs; the rest become remote (-1)."""
    ids=d['topk_ids'].clone();flat=ids.view(-1);local=flat-d['config']['local_start']
    for e in range(32):
        hit=((local>=0)&(local<32)&(local==e)).nonzero().flatten()
        if len(hit)>cap:flat[hit[cap:]]=-1
    return dict(d,topk_ids=ids)


def blocks(d):
    local=(d['topk_ids']-d['config']['local_start']).flatten();local=local[(local>=0)&(local<32)]
    n=torch.bincount(local.long(),minlength=32);return int(((n+7)//8).sum()),int((n>0).sum())


def lab(deadline,weights):
    from concurrent.futures import ThreadPoolExecutor
    begin=time.monotonic();deadline-=5;device=weights['w13'].device
    lines=['META small-M NT1: E=MoE node alone (fixed real routes), R=router node+MoE node; INC=admitted one-LOP3 NT1; F0=same math in the flag harness; Z=norm-slot zero in W13 tile0; SLOT=M1 top-8 slot dispatch; ZOUT=W2 stores v(1+w); RSQ=finalizer scale once per pair; NREG=norm sums in registers per tile run; LAST=last CTA finalizes; PDL=MoE launched as programmatic dependent',
        'META v10: cold = lab_cold_l2() (wrapper; judge-style read); paired AB/BA x6 per arm vs S11 map {1:95,8/16/32:69}, independent S11 A:A; 16 seeds of public weights/routers; TASK.md gate (1.5% L2, row 5x) per cell; WPUB (two writer warps publish without a CTA barrier), WPUB+RND, ALIGN (RND with parallel split search and per-round m-chunks), ALIGN+WPUB at M8/16/32, WPUB at M1; M32 base69 and WPUB critical paths with TILE (per-tile waits, publish fence) lines']
    errors=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={name:pool.submit(fn) for name,fn in (('INC',lambda:nt1_module('INC')),('VAR',var_module),('CONTROL',control_module),('CUPTI',activity_module))}
        for name,job in jobs.items():
            try:job.result()
            except Exception as e:errors.append(f'BUILD_ERROR {name} {type(e).__name__}: {str(e)[-200:]}')
    lines+=errors
    if any(name in e for e in errors for name in ('BUILD_ERROR INC','BUILD_ERROR VAR')):return lines
    lines.append(f'BUILD seconds={time.monotonic()-begin:.2f} left={deadline-time.monotonic():.1f}')
    rows={};deltas={};paired={};caps={};noise={m:{'E':[],'R':[]} for m in SMALLM_ARMS};worst={};failed={};flips=0;complete=0;peak=0.
    previous=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
    try:
        for seed in range(16):
            if deadline-time.monotonic()<max(8.,peak*1.25+2.):
                lines.append(f'BUDGET before_seed={seed}');break
            started=time.monotonic();w=rg_generate_weights(5353+7919*seed,device=device);w['post_norm_w']=w['post_expert_norm'];state=nt1_banks(w)
            if seed==0:
                for mt,cases in DIAG_BY_M.items():
                    try:lines.extend(f'M{mt} '+l for l in timeline_lines(timeline_diagnose(fx_make_fixture(w,mt,9001),state,routed_fixture(w,mt,9001),cases),False))
                    except Exception as e:lines.append(f'DIAG_ERROR M{mt} {type(e).__name__}: {str(e)[:200]}')
            for m,arms in SMALLM_ARMS.items():
                d=fx_make_fixture(w,m,9000+m+97*seed);want=nt1_fp32(d,w)
                pr,rs=routed_fixture(w,m,9000+m+97*seed);xr,score,bias=route_reference(pr,rs);start=int(rs['start'])
                graphs={};truths={}
                dcap=cap_fixture(d) if 'ECAP' in arms['E'] else None
                if dcap is not None:caps.setdefault(m,[]).append(blocks(d)+blocks(dcap))
                for label,flags in arms['E'].items():
                    if label in failed:continue
                    try:graphs[label]=TimelineGraph(lambda:nt1_call(d,state,'INC')) if flags is None else \
                        TimelineGraph(lambda:nt1_call(dcap,state,'INC')) if label=='ECAP' else TimelineGraph(lambda f=flags:var_call(d,state,f))
                    except Exception as e:failed[label]=f'M{m} seed{seed} capture {type(e).__name__}: {str(e)[:80]}'
                for label,flags in arms['R'].items():
                    if label in failed:continue
                    try:
                        graphs[label]=TimelineGraph(lambda:routed_experts(pr,rs,state,'INC')) if flags is None else \
                            TimelineGraph(lambda f=flags:routed_var(pr,rs,state,f))
                    except Exception as e:failed[label]=f'M{m} seed{seed} capture {type(e).__name__}: {str(e)[:80]}'
                def gate(label,first=False):
                    if label=='ECAP':return
                    out=graphs[label].replay();torch.cuda.synchronize()
                    if label[0]=='E':got,truth=out,want
                    else:
                        got,x,ids,tw,lids=out
                        n=route_gate(ids,bias,start)
                        if not torch.allclose(x.float(),xr.float(),rtol=.01,atol=.01):raise ValueError('pooled norm')
                        key=tuple(ids.flatten().cpu().tolist())
                        if key not in truths:truths[key]=nt1_fp32(dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0}),w)
                        truth=truths[key]
                        if label=='RBASE' and first:route_flips[0]+=n
                    ok,err,row=task_check(got,truth)
                    worst[label]=max(worst.get(label,(0.,0.)),(err,row))
                    if not ok:raise ValueError(f'TASK gate L2/row={err:.5f}/{row:.5f}')
                route_flips=[0]
                for label in list(graphs):
                    try:gate(label,True)
                    except Exception as e:
                        failed[label]=f'M{m} seed{seed} {type(e).__name__}: {str(e)[:100]}';graphs.pop(label)
                flips+=route_flips[0]
                for prefix in ('E','R'):
                    base=prefix+'BASE'
                    if base not in graphs:continue
                    ctl=TimelineGraph(lambda:nt1_call(d,state,'INC')) if prefix=='E' else TimelineGraph(lambda:routed_var(pr,rs,state,BASE_FLAGS[m]))
                    a,b,_=timeline_timing(graphs[base],ctl);noise[m][prefix].append(b-a);rows.setdefault((m,base),[]).append(a)
                    others=[k for k in graphs if k[0]==prefix and k!=base]
                    if seed%2:others.reverse()
                    for label in others:
                        a,b,ratio=timeline_timing(graphs[base],graphs[label])
                        paired.setdefault((m,label),[]).append(ratio)
                        rows.setdefault((m,label),[]).append(rows[m,base][-1]*ratio);deltas.setdefault((m,label),[]).append(rows[m,base][-1]*(ratio-1))
                        try:gate(label)
                        except Exception as e:failed[label]=f'M{m} seed{seed} post-timing {type(e).__name__}: {str(e)[:100]}'
                graphs={};truths={}
            complete+=1;peak=max(peak,time.monotonic()-started)
    finally:torch.backends.cuda.matmul.allow_tf32=previous
    for m,v in sorted(caps.items()):
        lines.append(f'CAP M{m} blocks/experts {statistics.mean(x[0] for x in v):.2f}/{statistics.mean(x[1] for x in v):.2f} -> {statistics.mean(x[2] for x in v):.2f}/{statistics.mean(x[3] for x in v):.2f} n={len(v)}')
    for label,why in failed.items():lines.append(f'FAIL {label} {why}')
    lines.append(f'GATE TASK.md L2/row worst '+','.join(f'{k}:{v[0]:.4f}/{v[1]:.4f}' for k,v in sorted(worst.items()))+f'; RINC near-tie flips={flips}')
    for m in SMALLM_ARMS:
        for prefix in ('E','R'):
            base=(m,prefix+'BASE')
            if base not in rows:continue
            nz=noise[m][prefix];parts=[f'{prefix}BASE{BASE_FLAGS[m]}={statistics.mean(rows[base]):.2f}']
            for label in SMALLM_ARMS[m][prefix]:
                if label==prefix+'BASE' or (m,label) not in deltas:continue
                dl=deltas[m,label];se=statistics.stdev(dl)/len(dl)**.5 if len(dl)>1 else 0.
                rat=paired[m,label];parts.append(f'{label}={statistics.mean(rows[m,label]):.2f}({statistics.mean(dl):+.2f}+-{se:.2f},r={statistics.mean(rat):.5f},win={sum(r<1 for r in rat)}/{len(rat)})')
            parts.append(f'AA={statistics.mean(nz):+.2f}/{statistics.mean(map(abs,nz)):.2f}/{max(map(abs,nz)):.2f}')
            lines.append(f'AGG M{m} '+' '.join(parts))
    lines.append(f'END seeds={complete} min8={complete>=8} seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}; nothing promoted here')
    return lines
