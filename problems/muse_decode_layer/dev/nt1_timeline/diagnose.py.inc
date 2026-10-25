def timeline_case(d,state,kind,flush,reps=4):
    ctl=control_module();dev=d['hidden_states'].device
    markers=torch.empty(4,device=dev,dtype=torch.int64)
    inc=nt1_module('INC');grid=min(264,state['sms']*inc.resident_blocks(1,2,len(d['hidden_states'])))
    smem=inc.smem_needed(1,2,len(d['hidden_states']))
    experts=(d['topk_ids']-d['config']['local_start']).flatten();experts=experts[(experts>=0)&(experts<32)].unique().int().contiguous()
    count=len(experts)
    if kind.startswith('STREAM') and not count:raise ValueError('stream diagnostic needs a routed expert')
    if kind.startswith('STREAM'):
        stamps=torch.empty((grid+16,4),device=dev,dtype=torch.int64)
        sums=torch.empty(((grid+16)*256,),device=dev,dtype=torch.int32)
    elif kind.startswith('EMPTY'):stamps=torch.empty((grid,16),device=dev,dtype=torch.int64)
    else:stamps=None
    positioned=kind!='INC_plain'
    def launch():
        nonlocal stamps
        stream=torch.cuda.current_stream(dev).cuda_stream
        if positioned:ctl.mark(markers.data_ptr(),0,stream)
        if kind.startswith('EMPTY'):
            ctl.empty(stamps.data_ptr(),1 if kind=='EMPTY_tiny' else grid,32 if kind=='EMPTY_tiny' else 256,0 if kind=='EMPTY_tiny' else smem,stream)
        elif kind.startswith('STREAM'):
            ctl.streaming(state['bank']['w13p'].data_ptr(),state['bank']['w2p'].data_ptr(),experts.data_ptr(),count,
                sums.data_ptr(),stamps.data_ptr(),grid,256,smem,stream)
        elif kind.startswith('TRACE'):
            out,stamps=nt1_call(d,state,kind,trace_out=True)
        else:out=nt1_call(d,state,'Z' if kind=='Z_positioned' else 'INC')
        if positioned:ctl.mark(markers.data_ptr(),1,stream)
    launch();torch.cuda.synchronize()
    ctl.profile_begin();samples=[]
    try:
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):launch()
        for _ in range(2):flush.zero_();graph.replay();torch.cuda.synchronize()
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for rep in range(reps):
            flush.zero_();a.record();graph.replay();b.record();b.synchronize()
            samples.append(dict(event_us=a.elapsed_time(b)*1000,markers=markers.cpu().tolist() if positioned else None,
                stamps=stamps.cpu().tolist() if stamps is not None else None))
    finally:
        torch.cuda.synchronize();records=ctl.profile_end()
    main_name='timed_moe' if kind.startswith('TRACE') else 'empty_like' if kind.startswith('EMPTY') else 'stream_banks' if kind.startswith('STREAM') else 'moe_fused'
    main=sorted([r for r in records if main_name in r[0] and r[1][4]>0],key=lambda r:r[1][0])[-reps:]
    before=sorted([r for r in records if 'node_before' in r[0]],key=lambda r:r[1][0])[-reps:]
    after=sorted([r for r in records if 'node_after' in r[0]],key=lambda r:r[1][0])[-reps:]
    if len(main)!=reps or (positioned and (len(before)!=reps or len(after)!=reps)):
        raise ValueError(f'incomplete CUPTI graph nodes {kind}: main={len(main)} before={len(before)} after={len(after)}')
    for i,s in enumerate(samples):
        rec=main[i][1];s['activity']=rec;s['node_us']=(rec[1]-rec[0])/1000
        if rec[0]<=0 or rec[1]<=rec[0]:raise ValueError('CUPTI node has no execution interval')
        if positioned:
            rb,ra=before[i][1],after[i][1];mb=s['markers'][0];ma=s['markers'][2]
            lo=max(rb[0]-mb,ra[0]-ma);hi=min(rb[1]-mb,ra[1]-ma)
            if lo>hi:raise ValueError('globaltimer/activity clock calibration has no intersection')
            s['offset_bounds']=[lo,hi];s['clock_uncertainty_us']=(hi-lo)/1000
            s['graph_span_us']=(ra[1]-rb[0])/1000
            s['graph_framing_us']=s['event_us']-s['graph_span_us']
            s['position_gaps_us']=((rec[0]-rb[1])+(ra[0]-rec[1]))/1000
        else:s['graph_framing_us']=s['event_us']-s['node_us']
    return dict(kind=kind,experts=experts.cpu().tolist(),unique_bytes=count*(18874368+9437184),grid=grid,profile_warmups=2,samples=samples)


def timeline_account(case):
    summaries=[]
    for s in case['samples']:
        rows=s['stamps'];entries=[min(r[32:40]) for r in rows];exits=[max(max(r[40:48]),r[9]) for r in rows]
        if not all(x>0 for x in entries+exits):raise ValueError('incomplete CTA/warp timestamp coverage')
        first=min(entries);last=max(exits);critical=max(range(len(rows)),key=lambda i:exits[i]);r=rows[critical]
        bounds=[entries[critical],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],exits[critical]]
        if any(a>b for a,b in zip(bounds,bounds[1:])):raise ValueError('non-monotone contiguous CTA boundaries')
        phases=[(b-a)/1000 for a,b in zip(bounds,bounds[1:])]
        ramp=(entries[critical]-first)/1000;span=(last-first)/1000;outside=s['node_us']-span
        if abs(ramp+sum(phases)+outside-s['node_us'])>1e-6:raise ValueError('node accounting does not close')
        lo,hi=s['offset_bounds'];node0,node1=s['activity'][:2]
        freq=[]
        for rr in rows:
            if rr[9]>rr[0]:freq.append((rr[21]-rr[20])*1000/(rr[9]-rr[0]))
        summaries.append(dict(node_us=s['node_us'],event_us=s['event_us'],graph_framing_us=s['graph_framing_us'],
            critical_cta=critical,critical_is_finalizer=bool(r[23]),ramp_to_critical_us=ramp,contiguous_phase_us=phases,
            cta_span_us=span,outside_cta_us=outside,clock_mhz=statistics.median(freq),
            first_dispatch_bounds_us=[(first+lo-node0)/1000,(first+hi-node0)/1000],
            exit_drain_bounds_us=[(node1-last-hi)/1000,(node1-last-lo)/1000],clock_uncertainty_us=s['clock_uncertainty_us'],
            entry_spread_us=(max(entries)-min(entries))/1000,
            after_last_W2_us=(last-max(rr[5] for rr in rows))/1000,
            finalizer_us=max((rr[8]-rr[7])/1000 for rr in rows if rr[23]),
            first_copy_to_data_us=statistics.median((rr[12]-rr[10])/1000 for rr in rows if rr[10] and rr[12])))
    return summaries


def timeline_diagnose(d,state,flush):
    cases={}
    for kind in ('INC_plain','INC_positioned','TRACE','Z_positioned','TRACEZ','EMPTY_match','EMPTY_tiny','STREAM'):
        cases[kind]=timeline_case(d,state,kind,flush)
    account=timeline_account(cases['TRACE'])
    return dict(cases=cases,account=account,account_z=timeline_account(cases['TRACEZ']),phase_order=['route','zero_init','W13','W2_first_ready','W2_rest','fence','grid_barrier','combine','observer_footer'])


def timeline_lines(result):
    cases=result['cases'];out=[]
    for k,c in cases.items():
        samples=c['samples'];node=statistics.median(s['node_us'] for s in samples);event=statistics.median(s['event_us'] for s in samples)
        out.append(f'NODE {k} event/node/frame_us={event:.3f}/{node:.3f}/{statistics.median(s["graph_framing_us"] for s in samples):.3f} graph_ids='+str(sorted(set(s['activity'][6] for s in samples)))+
            (f' unique_MB={c["unique_bytes"]/1e6:.3f} TBs={c["unique_bytes"]/node/1e6:.3f}' if k=='STREAM' else ''))
    acc=result['account'];med=lambda key:statistics.median(s[key] for s in acc)
    out.append(f'ACCOUNT TRACE node={med("node_us"):.3f} CTAspan={med("cta_span_us"):.3f} outside={med("outside_cta_us"):.3f} ramp={med("ramp_to_critical_us"):.3f} entry_spread={med("entry_spread_us"):.3f} MHz={med("clock_mhz"):.1f} tail_afterW2={med("after_last_W2_us"):.3f} finalizer={med("finalizer_us"):.3f} clock_bound_width={med("clock_uncertainty_us"):.3f}')
    out.append('CRITICAL route/zero/W13/firstReady/W2rest/fence/barrier/combine/footer_us='+','.join(f'{statistics.median(s["contiguous_phase_us"][i] for s in acc):.3f}' for i in range(9)))
    out.append('DISPATCH bounds_us='+str([round(statistics.median(s['first_dispatch_bounds_us'][i] for s in acc),3) for i in (0,1)])+' exit_drain_bounds_us='+str([round(statistics.median(s['exit_drain_bounds_us'][i] for s in acc),3) for i in (0,1)])+' observer_delta_us='+str(round(statistics.median(s['node_us'] for s in cases['TRACE']['samples'])-statistics.median(s['node_us'] for s in cases['INC_positioned']['samples']),3)))
    for label,rows in (('INC',result['account']),('Z',result['account_z'])):
        one=sorted(rows,key=lambda s:s['node_us'])[len(rows)//2]
        closed=one['outside_cta_us']+one['ramp_to_critical_us']+sum(one['contiguous_phase_us'])
        out.append('ONE '+label+f' node={one["node_us"]:.3f} outside={one["outside_cta_us"]:.3f} ramp={one["ramp_to_critical_us"]:.3f} phases='+','.join(f'{v:.3f}' for v in one['contiguous_phase_us'])+f' close_error={closed-one["node_us"]:.6f}')
    return out
