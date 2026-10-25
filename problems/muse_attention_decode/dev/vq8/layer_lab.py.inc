"""Whole-layer paired P5 port vs frozen final S11. A2 is a second identical control graph."""
def lab(deadline,weights,state):
    started=time.monotonic();dev=weights['qkv_gate'].device
    lines=['META final S11 whole-layer paired direct-partials port; 16 seeds x8 rounds, rotating graph order, read-flushed L2; ratio=variant/S11 (<1 wins)',
           'META A2=S11 duplicate, PP=direct-partials port with exact sincos; same static inputs/state/projection and MoE plans, TASK host correctness plus diagnostic A/B output check']
    # The exact original function from frozen final S11, bound to the control's own globals.
    a=ab_namespace({})
    exec(compile(P5_CONTROL_FUNCTION,'<p5_s11_attention>','exec'),a)
    b=ab_namespace({})
    results={};incomplete=[]
    for key in AB_CASES:
        if time.monotonic()>deadline-8:incomplete.append(ab_label(key));continue
        B,L,w=key
        if L!=1:continue
        lines.append(ab_plan_line(key,state))
        def mk(seed):
            return _ap_generate_input(batch=B,ctx=80000,layer='sliding' if w else 'full',q_len=L,
                spread='agentic' if B>1 else 'fixed',dist='soft',page=1,seed=seed,device=dev)
        data=mk(AB_SEEDS[0]);graphs={};gates={}
        for name,v in [('A',a),('A2',a),('PP',b)]:
            try:graphs[name]=ab_graph(lambda v=v:v['kernel_fn'](data,state))
            except Exception as e:gates[name]='capture:'+type(e).__name__+':'+str(e)[:100]
        if not all(n in graphs for n in ('A','A2','PP')):
            lines.append('CAPTURE '+ab_label(key)+' '+str(gates));continue
        seeds=[]
        for seed in AB_SEEDS:
            if time.monotonic()>deadline-4:incomplete.append(f'{ab_label(key)}@{seed}');break
            if seed!=AB_SEEDS[0]:ab_copy(data,mk(seed))
            for g,_ in graphs.values():g.replay()
            torch.cuda.synchronize()
            for name,(_,out) in graphs.items():
                if name!='A' and (reason:=ab_gate(out,graphs['A'][1])):gates.setdefault(name,reason)
            times=ab_time(graphs,AB_ROUNDS)
            seeds.append({n:statistics.mean(v) for n,v in times.items()})
        if seeds:results[key]=(ab_summary(seeds),gates)
        for n,(r,se,d,aus,k) in sorted(results.get(key,({},{}))[0].items()):
            lines.append(f'L {ab_label(key)} {n} r={r:.5f} se={se:.5f} d={d:+.2f} A={aus:.2f} n={k} '+gates.get(n,'PASS'))
        graphs=None;gc.collect();torch.cuda.empty_cache()
    for (B,L),label in AB_LOADS.items():
        keys=[(B,L,2048),(B,L,0)]
        if not all(k in results for k in keys):continue
        for name in ('A2','PP'):
            rs=[results[k][0][name] for k in keys]
            den=sum(w*r[3] for w,r in zip((46,16),rs));num=sum(w*r[3]*r[0] for w,r in zip((46,16),rs))
            lines.append(f'LOAD {label} {name} r={num/den:.5f}')
    if incomplete:lines.append('INCOMPLETE '+' '.join(incomplete))
    lines.append(f'END elapsed={time.monotonic()-started:.1f}s left={deadline-time.monotonic():.1f}s')
    return lines
