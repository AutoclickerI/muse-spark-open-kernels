PHASE_RESULTS=[]


def mechanism_stamp(st,t):
    out={}
    for role in ('b1','route','b2'):
        op=st['pg_ops'][(t,role)];rows=op.stamps.cpu().tolist()
        ready=max(r[0] for r in rows)
        # Maxima are a completion frontier, not one CTA's serial history.
        mx=[max(r[i] for r in rows) for i in range(8)]
        out[role]=dict(last_partial=0.,last_stats=(mx[3]-ready)/1000.,
            last_publish=(mx[4]-ready)/1000.,last_wait=(mx[1]-ready)/1000.,
            last_apply=(mx[5]-ready)/1000.,last_topk=(mx[6]-ready)/1000. if mx[6] else None,
            last_meta=(mx[7]-ready)/1000. if mx[7] else None,last_store=(mx[2]-ready)/1000.,
            native=type(op).__name__=='FixedOBoundary')
        # For native O, field5 is norm-parameters-ready; for Triton it is
        # apply-output-store-ready. Neither is a GPU node completion event.
    return out


def lab(deadline,weights,state=None):
    global PHASE_RESULTS
    PHASE_RESULTS=[]
    lines=['META mechanism of losing S11 294fe739 recipe; projections frozen; no promotion',
           'META CUPTI individual arms; PHASE us after last partial store [stats,publish,wait,apply,topk,meta,store]; maxima across CTAs; native apply=norm-ready, Triton apply=apply-stores-ready; stamps are diagnostics']
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(tl_activity);host,st=pg_host(weights);act=future.result()
    for b,q,w in ((1,1,2048),(16,1,2048),(32,1,2048),(1,8,2048)):
        if time.monotonic()>deadline-8:lines.append('INCOMPLETE');break
        t=b*q;label=('d' if q==1 else 'v')+str(b)+'S'
        data=host._ap_generate_input(batch=b,ctx=80000,layer='sliding',q_len=q,
            spread='fixed' if b==1 else 'agentic',dist='soft',page=1,seed=172903,device=weights['o'].device)
        arms={'A':dict(st,pg_roles=()),'B1':pg_state(host,st,t,('b1',)),
            'R':pg_state(host,st,t,('route',)), 'B2':pg_state(host,st,t,('b2',)),
            'F':pg_state(host,st,t,('b1','route','b2'),meta=False),
            'FM':pg_state(host,st,t,('b1','route','b2')),
            'RAW':pg_state(host,st,t,('b1','route','b2'),meta=False,fuse=False)}
        profiles={};graphs={};maps={};gates={}
        for n,arm in arms.items():
            try:
                for _ in range(2):host.kernel_fn(data,arm)
                torch.cuda.synchronize();act.profile_begin()
                try:
                    graph,out,mapping=pg_capture(host,arm,data,act)
                    graph.replay();torch.cuda.synchronize()
                finally:act.profile_end()
                graphs[n]=(graph,out);maps[n]=mapping
                if n!='A' and (why:=ab_gate(out,graphs['A'][1])):gates[n]=why
                profiles[n]=pg_profile(graph,mapping,act)
                lines.append(f'INC {label} {n}='+','.join(f'{v:.3f}' for v in pg_increments(profiles[n]))+
                    f' wall={profiles[n]["wall"]:.3f}')
            except Exception as exc:gates[n]=pg_error(exc)
        traced=pg_state(host,st,t,('b1','route','b2'),trace=True)
        try:
            gr,out=ab_graph(lambda:host.kernel_fn(data,traced));samples=[]
            for _ in range(5):
                lab_cold_l2();gr.replay();torch.cuda.synchronize();samples.append(mechanism_stamp(traced,t))
            if why:=ab_gate(out,graphs['A'][1]):gates['trace']=why
            phase={role:{k:statistics.median(v[role][k] for v in samples)
                if isinstance(samples[0][role][k],float) else samples[0][role][k]
                for k in samples[0][role]} for role in ('b1','route','b2')}
            lines.append('PHASE '+label+' '+json.dumps({role:[phase[role][k] for k in
                ('last_stats','last_publish','last_wait','last_apply','last_topk','last_meta','last_store')]
                for role in phase},separators=(',',':')))
        except Exception as exc:phase={};gates['trace']=pg_error(exc)
        compact={n:dict(wall=p['wall'],span_samples=p['span_samples'],stages=p['stages']) for n,p in profiles.items()}
        PHASE_RESULTS.append(dict(case=label,profiles=compact,phase=phase,gates=gates))
        if gates:lines.append('GATE '+label+' '+repr(gates))
        del graphs,arms,traced,data,profiles
        gc.collect();torch.cuda.empty_cache()
    lines.append('END shapes='+str(len(PHASE_RESULTS))+'/4')
    lines.append('PG_MECHANISM_XZ85 '+base64.b85encode(lzma.compress(json.dumps(PHASE_RESULTS,separators=(',',':')).encode(),preset=9)).decode())
    return lines
