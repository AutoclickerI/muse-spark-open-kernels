def fixed_choices(m):
    cs=[('R2','R2',2,264),('INC','INC',2,264),('L26','L26',2,264),
        ('A','A',2,264),('B','B',2,264),('C','C',2,264),('ABC','ABC',2,264),
        ('K2s2','K2',2,528),('K2s3','K2',3,528)]
    if m>=16:cs.extend([('D','D',2,264),('ABCD','ABCD',2,264),('D128','D128',2,264),('ABCD128','ABCD128',2,264)])
    return cs


def fixed_first_bad(got,want,d,stage):
    mask=(got!=want).reshape(-1).nonzero()
    if not mask.numel():return 'none'
    flat=int(mask[0]);p=flat//4096;col=flat%4096
    ids=d['topk_ids'].flatten();order=torch.argsort(ids,stable=True);order=order[(ids[order]>=0)&(ids[order]<32)]
    slot=int(order[p]);e=int(ids[slot])
    return f'{stage}:expert={e},pair={slot},row={p},col={col},got={float(got[p,col]):.7g},ref={float(want[p,col]):.7g}'


def lab(deadline,weights):
    from concurrent.futures import ThreadPoolExecutor
    deadline-=5 # wrapper allows285s; this payload leaves the host at<=280s
    begin=time.monotonic();device=weights['w13'].device
    lines=['META small-M fixed phases; R2/INC/L26, A=once-route, B=expert-local finish, C=timed W2 L2 lookahead, D=128 producer+256 consumer threads/double BF16 smem; K2=more CTA control',
           'META recoded NVFP4/u8 scales only; >=8 (target12) public router seeds M1/8/16/32; cold128MiB AB/BA each variant vs INC, independent INC A:A; FP32 gates and non-K2 exact inter/Y']
    with ThreadPoolExecutor(max_workers=3) as pool:
        for mod in pool.map(nt1_module,NT1_SOURCES):pass
    lines.append(f'BUILD seconds={time.monotonic()-begin:.2f} left={deadline-time.monotonic():.1f}')
    flush=torch.empty(128<<20,device=device,dtype=torch.uint8)
    rows={(m,c[0]):[] for m in (1,8,16,32) for c in fixed_choices(m)}
    aa={m:[] for m in (1,8,16,32)};worst={k:0. for k in NT1_SOURCES};peak=0.;complete=0
    previous=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
    try:
        for seed in range(12):
            if deadline-time.monotonic()<max(6.,peak*1.3+2.):
                lines.append(f'BUDGET before_seed={seed}');break
            started=time.monotonic();w=rg_generate_weights(5353+7919*seed,device=device);w['post_norm_w']=w['post_expert_norm']
            state=nt1_banks(w)
            for m in (1,8,16,32):
                d=fx_make_fixture(w,m,9000+m);want=nt1_fp32(d,w)
                ref=nt1_call(d,state,'INC',scratch=True);torch.cuda.synchronize()
                p=int(((d['topk_ids']>=0)&(d['topk_ids']<32)).sum())
                graphs={};choices=fixed_choices(m)
                for label,kind,st,grid in choices:
                    graph=LabGraph(lambda d,k=kind,s=st,g=grid:nt1_call(d,state,k,s,g),d)
                    got=graph.replay();torch.cuda.synchronize();err,row=nt1_error(got,want);worst[kind]=max(worst[kind],err)
                    if not bool(torch.isfinite(got).all()) or err>.01 or row>.05:
                        ix=int((got.float()-want).abs().flatten().argmax())
                        return lines+[f'REJECT {label} M{m} seed{seed} L2/row={err:.6f}/{row:.6f} token={ix//4096} col={ix%4096}']
                    if kind!='K2':
                        debug=nt1_call(d,state,kind,scratch=True);torch.cuda.synchronize()
                        for j,name in ((1,'SwiGLU'),(2,'W2')):
                            if not torch.equal(debug[j][:p],ref[j][:p]):
                                return lines+['REJECT '+label+' M'+str(m)+' seed'+str(seed)+' '+fixed_first_bad(debug[j][:p],ref[j][:p],d,name)]
                    graphs[label]=graph
                aa_graph=LabGraph(lambda d:nt1_call(d,state,'INC'),d)
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                def measured(g):
                    flush.zero_();a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
                aat=[];inc_samples=[]
                for rep in range(4):
                    order=(graphs['INC'],aa_graph) if rep%2==0 else (aa_graph,graphs['INC'])
                    ts=[measured(g) for g in order]
                    ta,tb=ts if rep%2==0 else ts[::-1]
                    aat.append(tb-ta);inc_samples.append(ta)
                aa[m].extend(aat);base=statistics.median(inc_samples);rows[m,'INC'].append(base)
                others=[c[0] for c in choices if c[0]!='INC']
                if seed%2:others.reverse()
                for label in others:
                    ratios=[]
                    for rep in range(4):
                        order=(graphs['INC'],graphs[label]) if rep%2==0 else (graphs[label],graphs['INC'])
                        ts=[measured(g) for g in order];ta,tb=ts if rep%2==0 else ts[::-1];ratios.append(tb/ta)
                    rows[m,label].append(base*statistics.median(ratios))
                    got=graphs[label].replay();torch.cuda.synchronize();e,r=nt1_error(got,want)
                    if e>.01 or r>.05:return lines+[f'REJECT after timing {label} M{m} seed{seed} {e}/{r}']
                if seed==0:
                    lines.append('GRID M'+str(m)+' '+','.join(f'{l}:{s}/{min(g,state["sms"]*nt1_module(k).resident_blocks(1,s,m))}' for l,k,s,g in choices))
                graphs={};graph=aa_graph=None;ref=None
            complete+=1;peak=max(peak,time.monotonic()-started)
    finally:torch.backends.cuda.matmul.allow_tf32=previous
    lines.append('GATE maxFP32='+','.join(f'{k}:{v:.5f}' for k,v in worst.items())+' exact inter/Y all non-K2 PASS; AA=independent captures')
    for m in (1,8,16,32):
        names=[c[0] for c in fixed_choices(m)];n=len(rows[m,'INC']);noise=aa[m]
        vals=','.join(f'{statistics.mean(rows[m,k]):.2f}' if rows[m,k] else '-' for k in names)
        lines.append(f'AGG M{m} n={n}/12 order='+','.join(names)+' mean_us='+vals+
            (f' AA_delta/abs/max={statistics.mean(noise):.2f}/{statistics.mean(map(abs,noise)):.2f}/{max(map(abs,noise)):.2f}' if noise else ''))
    return lines+[f'END seeds={complete}/12 min8_required={complete>=8} seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}; no variant promoted']
