# S11 functions and the reuse cell are frozen above this driver; fixture/runtime definitions are inherited.
def reuse_routed(pr,rs,state,mode):
    x,ids,tw,lids=route_call(pr,rs)
    d=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0})
    out=smallm_experts(d,state,69) if mode<0 else reuse_experts(d,state,mode)
    return out,x,ids,tw,lids


def reuse_gate(output,w,truths,pr,rs):
    got,x,ids,tw,lids=output;xr,_,bias=route_reference(pr,rs)
    route_gate(ids,bias,int(rs['start']))
    if not torch.allclose(x.float(),xr.float(),rtol=.01,atol=.01):raise ValueError('pooled norm')
    key=tuple(ids.flatten().cpu().tolist())
    if key not in truths:
        truths[key]=nt1_fp32(dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0}),w)
    ok,err,row=task_check(got,truths[key])
    if not ok:raise ValueError(f'TASK gate {err:.6f}/{row:.6f}')
    return err,row


def reuse_trace_lines(pr,rs,state):
    lines=[]
    for mode in (0,1):
        x,ids,tw,lids=route_call(pr,rs);m=x.shape[0];grid,*_=reuse_geometry(state,m,mode,True)
        stamps=torch.zeros((grid,16),device=x.device,dtype=torch.int64)
        d=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0})
        g=TimelineGraph(lambda:reuse_experts(d,state,mode,True,stamps));g.replay();torch.cuda.synchronize()
        rsamp=stamps.cpu().tolist();live=[r for r in rsamp if r[10]+r[11]]
        names=('route','copy_wait','decode','MMA_activation','reduce_epi','publish','ready','grid_barrier')
        parts=[]
        for slot,name in enumerate(names,1):parts.append(name+'='+str(round(statistics.median(r[slot] for r in live))))
        lines.append(f'TRACE M{m} '+('REDO' if mode else 'ONCE')+' clocks_per_CTA '+','.join(parts)+
            f' span_med/max={statistics.median(r[9]-r[0] for r in live):.0f}/{max(r[9]-r[0] for r in live)}'+
            f' units13/2={sum(r[10] for r in live)}/{sum(r[11] for r in live)}'+
            f' blockunits13/2={sum(r[12] for r in live)}/{sum(r[13] for r in live)}')
    return lines


def lab(deadline,weights):
    from concurrent.futures import ThreadPoolExecutor
    begin=time.monotonic();device=next(v.device for v in weights.values() if isinstance(v,torch.Tensor));deadline-=5
    lines=['META P3a unique weight-tile ownership; ONCE decodes each K group once per expert over all token blocks; REDO fixed-work re-decode control, same ownership, activation streaming, MMA and epilogues. S11=exact frozen acda6c62 smallm flags69.',
        'META router+MoE CUDA graphs; judge-style read flush; AB/BA x6 per arm per seed; 16 independent weight/router seeds; TASK gate per arm before and after timing. r=new/S11; ONCE/REDO reuses each paired seed ratio. TRACE is separately compiled, clock64 cycles, intrusive, not an additive attribution. No persistent decoded bank or dropped work.']
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(smallm_module),pool.submit(reuse_module)]
        for j in jobs:j.result()
    lines.append(f'BUILD seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}')
    ratios={};deltas={};times={};aa={m:[] for m in (8,16,32)};fails={};worst={};counts={m:[] for m in (8,16,32)};done=0;peak=0.
    oldtf=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
    try:
        for seed in range(16):
            if deadline-time.monotonic()<max(8,peak*1.15+2):lines.append(f'BUDGET before_seed={seed}');break
            started=time.monotonic();w=rg_generate_weights(5353+7919*seed,device=device);w['post_norm_w']=w['post_expert_norm'];base=nt1_banks(w)
            state={k:base[k] for k in ('pnw','gs13','gs2','sms')};state['banks']={'lop3':base['bank']}
            smallm_bind(state,{8:69,16:69,32:69});reuse_bind(state)
            if seed==0:
                for mode in (0,1):
                    for trace in (0,1):lines.append(f'GEOM mode={mode} trace={trace} grid,regs,localB,smemB,resident='+str(reuse_geometry(state,32,mode,trace)))
            for m in (8,16,32):
                pr,rs=routed_fixture(w,m,9000+m+97*seed);truths={};graphs={}
                x,ids,tw,lids=route_call(pr,rs)
                cnt=torch.bincount(lids[lids>=0].long(),minlength=32);counts[m].append((int((cnt>0).sum()),int(((cnt+7)//8).sum())))
                for name,mode in (('S11',-1),('ONCE',0),('REDO',1)):
                    if name in fails:continue
                    try:
                        graphs[name]=TimelineGraph(lambda mode=mode:reuse_routed(pr,rs,state,mode))
                        worst[name]=max(worst.get(name,(0.,0.)),reuse_gate(graphs[name].replay(),w,truths,pr,rs))
                    except Exception as exc:fails[name]=f'M{m} seed{seed} {type(exc).__name__} {str(exc)[:140]}';graphs.pop(name,None)
                if 'S11' not in graphs:continue
                control=TimelineGraph(lambda:reuse_routed(pr,rs,state,-1))
                a,b,_=timeline_timing(graphs['S11'],control);aa[m].append(b-a);times.setdefault((m,'S11'),[]).append(a)
                labels=['ONCE','REDO'] if seed%2==0 else ['REDO','ONCE']
                for name in labels:
                    if name not in graphs:continue
                    a,b,r=timeline_timing(graphs['S11'],graphs[name]);ratios.setdefault((m,name),[]).append(r)
                    times.setdefault((m,name),[]).append(b);deltas.setdefault((m,name),[]).append(b-a)
                    try:worst[name]=max(worst.get(name,(0.,0.)),reuse_gate(graphs[name].replay(),w,truths,pr,rs))
                    except Exception as exc:fails[name]=f'M{m} seed{seed} after {type(exc).__name__} {str(exc)[:140]}'
                if 'ONCE' in graphs and 'REDO' in graphs:
                    a,b,r=timeline_timing(graphs['REDO'],graphs['ONCE']);ratios.setdefault((m,'ONCE/REDO'),[]).append(r)
                    deltas.setdefault((m,'ONCE/REDO'),[]).append(b-a)
                if seed==0 and m==32:
                    try:lines.extend(reuse_trace_lines(pr,rs,state))
                    except Exception as exc:lines.append('TRACE_ERROR '+str(exc)[:160])
                graphs={};truths={}
            done+=1;peak=max(peak,time.monotonic()-started)
    finally:torch.backends.cuda.matmul.allow_tf32=oldtf
    for name,why in fails.items():lines.append(f'FAIL {name} {why}')
    lines.append('GATE '+','.join(f'{k}:{v[0]:.5f}/{v[1]:.5f}' for k,v in sorted(worst.items())))
    for m in (8,16,32):
        if not aa[m]:continue
        parts=[f'S11_us={statistics.mean(times[m,"S11"]):.2f}']
        for name in ('ONCE','REDO','ONCE/REDO'):
            r=ratios.get((m,name),[])
            if not r:continue
            se=statistics.stdev(r)/len(r)**.5 if len(r)>1 else 0.
            parts.append(f'{name} r={statistics.mean(r):.5f} se={se:.5f} d_us={statistics.mean(deltas[m,name]):+.2f} wins={sum(x<1 for x in r)}/{len(r)}'+
                (f' us={statistics.mean(times[m,name]):.2f}' if (m,name) in times else ''))
        es=statistics.mean(x[0] for x in counts[m]);bs=statistics.mean(x[1] for x in counts[m])
        parts.append(f'experts/blocks={es:.2f}/{bs:.2f} AA={statistics.mean(aa[m]):+.2f}/{statistics.mean(map(abs,aa[m])):.2f}/{max(map(abs,aa[m])):.2f}')
        lines.append(f'AGG M{m} '+' | '.join(parts))
    lines.append(f'END seeds={done} seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}; no promotion before paired whole-layer test')
    return lines
