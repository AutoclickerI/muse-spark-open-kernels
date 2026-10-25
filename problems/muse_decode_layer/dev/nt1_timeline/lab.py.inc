class TimelineGraph:
    def __init__(self,fn):
        fn();torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.out=fn()
    def replay(self):self.graph.replay();return self.out


def timeline_timing(old,new,flush,repeats=6):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    aa,bb,ratios=[],[],[]
    def measured(graph):
        flush.zero_();a.record();graph.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
    for rep in range(repeats):
        order=(old,new) if rep%2==0 else (new,old);v=[measured(g) for g in order]
        x,y=v if rep%2==0 else v[::-1];aa.append(x);bb.append(y);ratios.append(y/x)
    return statistics.median(aa),statistics.median(bb),statistics.median(ratios),[b-a for a,b in zip(aa,bb)]


def timeline_gate(out,want,label,m,seed):
    err,row=nt1_error(out,want)
    if not bool(torch.isfinite(out).all()) or err>.01 or row>.05:
        token=int((out.float()-want).norm(dim=-1).argmax());raise ValueError(f'{label} M{m} seed{seed}: FP32 L2/row={err:.6f}/{row:.6f} token={token}')
    return err,row


def lab(deadline,weights):
    from concurrent.futures import ThreadPoolExecutor
    begin=time.monotonic();deadline-=5;device=weights['w13'].device
    lines=['META complete graph-node/CTA globaltimer accounting; INC=accepted one-LOP3 recoded NT1; Z=norm-slot zero in W13 before expert release; no extra launch',
        'META E=fixed routes, R=router+experts (2 nodes); RMETA=router final-CTA emits shared v1, RMETAZ=both; cold128MiB paired AB/BA + independent A:A, target12/min8 router seeds, official near-tie gates']
    with ThreadPoolExecutor(max_workers=3) as pool:
        for mod in pool.map(nt1_module,NT1_SOURCES):pass
    control_module()
    lines.append(f'BUILD seconds={time.monotonic()-begin:.2f} left={deadline-time.monotonic():.1f}')
    flush=torch.empty(128<<20,device=device,dtype=torch.uint8)
    labels=('EINC','EZ','RINC','RZ','RMETA','RMETAZ');rows={(m,k):[] for m in (1,8,16,32) for k in labels}
    noise={(m,k):[] for m in (1,8,16,32) for k in ('E','R')};worst=0.;rowmax=0.;flips=0;paircounts={m:[] for m in (1,8,16,32)}
    complete=0;peak=0.;previous=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
    try:
        for seed in range(12):
            if deadline-time.monotonic()<max(5.,peak*1.3+1.):
                lines.append(f'BUDGET before_seed={seed}');break
            started=time.monotonic();w=rg_generate_weights(5353+7919*seed,device=device);w['post_norm_w']=w['post_expert_norm'];state=nt1_banks(w)
            if seed==0:
                # Same real-router two-expert fixture as the fixed-route timing.
                d0=fx_make_fixture(w,1,9001)
                diag=timeline_diagnose(d0,state,flush);lines.extend(timeline_lines(diag))
                for kind in NT1_SOURCES:
                    mod=nt1_module(kind);lines.append(f'GRID {kind} M1 grid={min(264,state["sms"]*mod.resident_blocks(1,2,1))} stages=2 smem={mod.smem_needed(1,2,1)}')
            for m in (1,8,16,32):
                d=fx_make_fixture(w,m,9000+m);want=nt1_fp32(d,w)
                pr,rs=routed_fixture(w,m,9000+m);xr,score,bias=route_reference(pr,rs);start=int(rs['start'])
                graphs={'EINC':TimelineGraph(lambda:nt1_call(d,state,'INC')),'EZ':TimelineGraph(lambda:nt1_call(d,state,'Z'))}
                for label,kind,emit in (('RINC','INC',False),('RZ','Z',False),('RMETA','META',True),('RMETAZ','METAZ',True)):
                    graphs[label]=TimelineGraph(lambda k=kind,e=emit:routed_experts(pr,rs,state,k,e))
                truths={};ids_seen={}
                for label,g in graphs.items():
                    out=g.replay();torch.cuda.synchronize()
                    if label.startswith('E'):got=out;truth=want
                    else:
                        got,x,ids,tw,lids,meta=out
                        flips+=route_gate(ids,bias,start)
                        if not torch.allclose(x.float(),xr.float(),rtol=.01,atol=.01):raise ValueError(f'{label} pooled norm M{m} seed{seed}')
                        expected=score.gather(1,ids.long());expected=(expected/expected.sum(-1,keepdim=True)).float()
                        if not torch.allclose(tw,expected,rtol=.001,atol=1e-5):raise ValueError('router weights')
                        # Each arm is judged on its own legal picks; near-tie flips are retained.
                        key=tuple(ids.flatten().cpu().tolist())
                        if key not in truths:
                            rd=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0});truths[key]=nt1_fp32(rd,w)
                        truth=truths[key];ids_seen[label]=ids
                    e,r=timeline_gate(got,truth,label,m,seed);worst=max(worst,e);rowmax=max(rowmax,r)
                paircounts[m].append(int((d['topk_ids']>=0).sum()))
                # Independent captures for both graph positions establish the noise floor.
                controls={'E':TimelineGraph(lambda:nt1_call(d,state,'INC')),
                    'R':TimelineGraph(lambda:routed_experts(pr,rs,state,'INC',False))}
                for prefix in ('E','R'):
                    base=graphs[prefix+'INC'];a,b,ratio,deltas=timeline_timing(base,controls[prefix],flush)
                    noise[m,prefix].extend(deltas);rows[m,prefix+'INC'].append(a)
                others=['EZ','RZ','RMETA','RMETAZ']
                if seed%2:others.reverse()
                for label in others:
                    baseline=label[0]+'INC';a,b,ratio,_=timeline_timing(graphs[baseline],graphs[label],flush)
                    rows[m,label].append(rows[m,baseline][-1]*ratio)
                    out=graphs[label].replay();torch.cuda.synchronize()
                    if label.startswith('E'):got,truth=out,want
                    else:
                        got=out[0];truth=truths[tuple(out[2].flatten().cpu().tolist())]
                    timeline_gate(got,truth,label,m,seed)
                graphs={};controls={};truths={}
            complete+=1;peak=max(peak,time.monotonic()-started)
    finally:torch.backends.cuda.matmul.allow_tf32=previous
    lines.append(f'GATE FP32maxL2/row={worst:.6f}/{rowmax:.6f} legal_neartie_token_flips={flips}; no dispatch promoted')
    # Keep summaries last so the wrapper's 9k-character retention preserves them.
    for m in (1,8,16,32):
        vals=','.join(f'{statistics.mean(rows[m,k]):.2f}' if rows[m,k] else '-' for k in labels)
        ns=','.join(prefix+':'+('/'.join(f'{v:.2f}' for v in (statistics.mean(noise[m,prefix]),statistics.mean(map(abs,noise[m,prefix])),max(map(abs,noise[m,prefix]))))) for prefix in ('E','R')) if complete else '-'
        lines.append(f'AGG M{m} n={complete}/12 labels='+','.join(labels)+' mean_us='+vals+f' AA_signed/abs/max={ns} pairs_mean={statistics.mean(paircounts[m]) if complete else 0:.1f}')
    lines.append(f'END min8_required={complete>=8} seeds={complete}/12 seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}')
    return lines
