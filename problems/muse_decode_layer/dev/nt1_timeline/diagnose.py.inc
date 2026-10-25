DIAG_ALL=F_Z|F_SLOT|F_ZOUT|F_RSQ|F_NREG
DIAG_ALLM=F_Z|F_ZOUT|F_RSQ|F_NREG
DIAG_FLAGS={'TRACE0':F_TRACE,'TRACEZZP':F_TRACE|F_Z|F_ZOUT|F_PDL,'TRACEALL':F_TRACE|DIAG_ALL|F_LAST,'TRACEALLnl':F_TRACE|DIAG_ALL,'ALL_positioned':DIAG_ALL|F_LAST,
    'RTRACE0':F_TRACE,'RTRACEPDL':F_TRACE|F_PDL,'RTRACEALLnl':F_TRACE|DIAG_ALL|F_PDL,'RTRACEALL':F_TRACE|DIAG_ALL|F_LAST|F_PDL,
    'TRACEALLM':F_TRACE|DIAG_ALLM,'RTRACEALLM':F_TRACE|DIAG_ALLM|F_PDL,'RTRACEZZP':F_TRACE|F_Z|F_ZOUT|F_PDL,'RTRACEZZP2':F_TRACE|F_Z|F_ZOUT|F_PDL|F_NT2,'RTRACEZZPW':F_TRACE|F_Z|F_ZOUT|F_PDL|F_NT2|F_W16,
    'RTRACEZZPR':F_TRACE|F_Z|F_ZOUT|F_PDL|F_RND,'RTRACEZZPO':F_TRACE|F_Z|F_ZOUT|F_PDL|F_ROT,
    'RTRACEZZPRO':F_TRACE|F_Z|F_ZOUT|F_PDL|F_RND|F_ROT}
DIAG_CASES=('INC_plain','INC_positioned','TRACE0','TRACEALL','TRACEALLnl','ALL_positioned','EMPTY_match','EMPTY_tiny','STREAM',
    'RTRACE0','RTRACEPDL','RTRACEALLnl','RTRACEALL')
DIAG_CASES_M=('INC_plain','TRACE0','TRACEALLM','STREAM','RTRACE0','RTRACEALLM','RTRACEZZP')


def timeline_case(d,state,kind,reps=10,cupti=True,routed=None):
    """One graph: [marker] node [marker]; event time, marker stamps, per-CTA stamps, optional CUPTI node records."""
    ctl=control_module();dev=d['hidden_states'].device;m=len(d['hidden_states'])
    markers=torch.zeros(4,device=dev,dtype=torch.int64)
    inc=nt1_module('INC');grid=min(264,state['sms']*inc.resident_blocks(1,2,m));smem=inc.smem_needed(1,2,m)
    experts=(d['topk_ids']-d['config']['local_start']).flatten();experts=experts[(experts>=0)&(experts<32)].unique().int().contiguous()
    count=len(experts)
    flags=DIAG_FLAGS.get(kind)
    if flags is not None and flags&F_NT2:grid=var_geometry(state,flags,m)[2]
    if kind=='STREAM':
        if not count:raise ValueError('stream control needs a routed expert')
        stamps=torch.zeros((grid+16,4),device=dev,dtype=torch.int64);sums=torch.empty(((grid+16)*256,),device=dev,dtype=torch.int32)
    elif kind.startswith('EMPTY'):stamps=torch.zeros((grid,16),device=dev,dtype=torch.int64)
    elif 'TRACE' in kind:stamps=torch.zeros((grid,64),device=dev,dtype=torch.int64)
    else:stamps=None
    positioned=not kind.startswith('INC_plain')
    def launch():
        stream=torch.cuda.current_stream(dev).cuda_stream
        if positioned:ctl.mark(markers.data_ptr(),0,stream)
        if kind.startswith('EMPTY'):
            ctl.empty(stamps.data_ptr(),1 if kind=='EMPTY_tiny' else grid,32 if kind=='EMPTY_tiny' else 256,0 if kind=='EMPTY_tiny' else smem,stream)
        elif kind=='STREAM':
            ctl.streaming(state['bank']['w13p'].data_ptr(),state['bank']['w2p'].data_ptr(),experts.data_ptr(),count,
                sums.data_ptr(),stamps.data_ptr(),grid,256,smem,stream)
        elif kind[0]=='R':routed_var(*routed,state,flags,stamps=stamps)
        elif flags is not None:var_call(d,state,flags,stamps=stamps)
        else:nt1_call(d,state,'INC')
        if positioned:ctl.mark(markers.data_ptr(),1,stream)
    launch();torch.cuda.synchronize()
    act=None
    if cupti:
        try:act=activity_module();act.profile_begin()
        except Exception as e:act=None;cupti=f'{type(e).__name__}: {e}'[:120]
    samples=[]
    try:
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):launch()
        for _ in range(2):lab_cold_l2();graph.replay();torch.cuda.synchronize()
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for rep in range(reps):
            lab_cold_l2();a.record();graph.replay();b.record();b.synchronize()
            samples.append(dict(event_us=a.elapsed_time(b)*1000,markers=markers.cpu().tolist() if positioned else None,
                stamps=stamps.cpu().tolist() if stamps is not None else None))
    finally:
        torch.cuda.synchronize();records=act.profile_end() if act is not None else None
    if records is not None:
        main_name='moe_var' if flags is not None else 'empty_like' if kind.startswith('EMPTY') else 'stream_banks' if kind=='STREAM' else 'moe_fused'
        main=sorted([r for r in records if main_name in r[0] and r[1][4]>0],key=lambda r:r[1][0])[-reps:]
        before=sorted([r for r in records if 'node_before' in r[0]],key=lambda r:r[1][0])[-reps:]
        after=sorted([r for r in records if 'node_after' in r[0]],key=lambda r:r[1][0])[-reps:]
        router=sorted([r for r in records if 'timeline_route' in r[0]],key=lambda r:r[1][0])[-reps:]
        if len(main)!=reps or (positioned and (len(before)!=reps or len(after)!=reps)):records=None;cupti='incomplete records'
    for i,s in enumerate(samples):
        if positioned:s['marker_span_us']=(s['markers'][2]-s['markers'][0])/1000
        if records is None:continue
        rec=main[i][1];s['activity']=rec;s['node_us']=(rec[1]-rec[0])/1000
        if len(router)==reps:s['router_us']=(router[i][1][1]-router[i][1][0])/1000;s['router_end_to_moe_start_us']=(rec[0]-router[i][1][1])/1000
        if positioned:
            rb,ra=before[i][1],after[i][1];mb=s['markers'][0];ma=s['markers'][2]
            lo=max(rb[0]-mb,ra[0]-ma);hi=min(rb[1]-mb,ra[1]-ma)
            if lo<=hi:s['offset_bounds']=[lo,hi]
            s['graph_span_us']=(ra[1]-rb[0])/1000
    return dict(kind=kind,experts=experts.cpu().tolist(),unique_bytes=count*(18874368+9437184),grid=grid,samples=samples,cupti=cupti)


def skew_stats(rows,first):
    """W13 end, W2 end and grid-barrier exit per CTA (us from the first entry) and units per CTA: where W2 skew sits."""
    def q(v):
        v=sorted(v);return (v[len(v)//2],v[(9*len(v))//10],v[-1]) if v else (0.,0.,0.)
    w13=[(r[3]-first)/1000 for r in rows if r[3] and r[26]];w2=[(r[5]-first)/1000 for r in rows if r[5] and r[27]]
    bar=[(r[7]-first)/1000 for r in rows if r[7]];u1=[r[26] for r in rows];u2=[r[27] for r in rows]
    return q(w13)+q(w2)+q(bar)+(min(u1),max(u1),min(u2),max(u2))


def tile_stats(rows,first):
    """Where a CTA's tile time goes (per-CTA sums, us; median over CTAs): tid 0 (epilogue warp) and tid 224 (MMA-only warp)
    waits on landed tiles and at the per-tile barrier, the expert publish (fence, then barrier+atomic), the W13 and W2 phase
    per tile, and how late the last publish lands after the last W13 end."""
    def p50(v):
        v=sorted(v);return v[len(v)//2] if v else 0.
    w=[r for r in rows if r[26]];w2=[r for r in rows if r[27]]
    return dict(
        wait13_t0=p50(r[28]/1000 for r in w),wait2_t0=p50(r[29]/1000 for r in w2),bar_t0=p50(r[22]/1000 for r in rows),
        bar_w7=p50(r[30]/1000 for r in rows),wait_w7=p50(r[31]/1000 for r in rows),
        fence=p50(r[17]/1000 for r in w),fence_max=max((r[17]/1000 for r in w),default=0.),
        pubbar=p50(r[19]/1000 for r in w),pubbar_max=max((r[19]/1000 for r in w),default=0.),
        per13=p50((r[3]-r[2])/1000/r[26] for r in w if r[3]),per2=p50((r[5]-r[4])/1000/r[27] for r in w2 if r[5] and r[4]),
        last_pub_after_last13=(max(r[18] for r in w if r[18])-max(r[3] for r in w))/1000 if any(r[18] for r in w) else 0.,
        pub_after_own13=p50((r[18]-r[3])/1000 for r in w if r[18] and r[3]))


def timeline_account(case):
    """Critical CTA (latest exit) as contiguous phases, plus the dispatch ramp and drain against the marker stamps."""
    out=[]
    for s in case['samples']:
        rows=[r for r in s['stamps'] if r[0]];entries=[min(r[32:40]) for r in rows];exits=[max(max(r[40:48]),r[9]) for r in rows]
        first=min(entries);last=max(exits);k=max(range(len(rows)),key=lambda i:exits[i]);r=rows[k]
        cut=[entries[k]]+[r[i] if r[i] else 0 for i in (1,2,3,4,5,6,7,8)]+[exits[k]]
        for i in range(1,len(cut)):cut[i]=cut[i] or cut[i-1]
        phases=[(b-a)/1000 for a,b in zip(cut,cut[1:])]
        res=dict(event_us=s['event_us'],cta_span_us=(last-first)/1000,ramp_to_critical_us=(entries[k]-first)/1000,phases=phases,
            entry_spread_us=(max(entries)-first)/1000,critical_finalizer=bool(r[23]),
            launch_us=(first-s['markers'][0])/1000 if s.get('markers') else None,
            drain_us=(s['markers'][2]-last)/1000 if s.get('markers') else None,
            marker_span_us=s.get('marker_span_us'),node_us=s.get('node_us'),
            after_last_W2_us=(last-max(rr[5] for rr in rows))/1000,
            last_W13_end_us=(max(rr[3] for rr in rows)-first)/1000,
            first_copy_issue_us=statistics.median((rr[10]-min(rr[32:40]))/1000 for rr in rows if rr[10]),
            first_copy_land_us=statistics.median((rr[12]-rr[10])/1000 for rr in rows if rr[10] and rr[12]),
            w2_wait_med_us=statistics.median(rr[16]/1000 for rr in rows if rr[27]),
            w2_wait_max_us=max(rr[16]/1000 for rr in rows if rr[27]),
            skew=skew_stats(rows,first),tile=tile_stats(rows,first))
        if s.get('offset_bounds') and s.get('activity'):
            lo,hi=s['offset_bounds'];n0,n1=s['activity'][:2]
            res['node_to_first_cta_us']=[(first+lo-n0)/1000,(first+hi-n0)/1000];res['last_cta_to_node_end_us']=[(n1-last-hi)/1000,(n1-last-lo)/1000]
        out.append(res)
    return out


def timeline_diagnose(d,state,routed,cases_run=DIAG_CASES):
    ctl=control_module();res=torch.zeros(3,device=d['hidden_states'].device,dtype=torch.int64)
    ctl.resolution(res.data_ptr(),torch.cuda.current_stream().cuda_stream);torch.cuda.synchronize();plain=res.cpu().tolist()
    cases={}
    for kind in cases_run:
        try:cases[kind]=timeline_case(d,state,kind,routed=routed)
        except Exception as e:cases[kind]=dict(kind=kind,error=f'{type(e).__name__}: {e}'[:160])
    profiled=None
    try:
        act=activity_module();act.profile_begin()
        try:ctl.resolution(res.data_ptr(),torch.cuda.current_stream().cuda_stream);torch.cuda.synchronize();profiled=res.cpu().tolist()
        finally:act.profile_end()
    except Exception as e:profiled=f'{type(e).__name__}'
    acc={k:timeline_account(c) for k,c in cases.items() if 'TRACE' in k and 'samples' in c}
    return dict(cases=cases,account=acc,resolution=dict(plain=plain,profiled=profiled))


def timeline_lines(result,resolution=True):
    out=[];r=result['resolution']
    if resolution:out.append(f'GTRES plain_min/changes/span_ns={r["plain"]} under_cupti={r["profiled"]}')
    med=statistics.median
    for k,c in result['cases'].items():
        if 'error' in c:out.append(f'NODE {k} ERROR {c["error"]}');continue
        if k in result['account'] and not k.startswith('R'):continue  # its ACCOUNT line carries event/marker/node
        s=c['samples'];line=f'NODE {k} event={med(x["event_us"] for x in s):.2f}'
        if s[0].get('marker_span_us') is not None:line+=f' marker_span={med(x["marker_span_us"] for x in s):.2f}'
        if s[0].get('node_us') is not None:line+=f' cupti_node={med(x["node_us"] for x in s):.2f}'
        elif c.get('cupti') not in (True,None):line+=f' cupti={c["cupti"]}'
        if s[0].get('router_us') is not None:line+=f' router={med(x["router_us"] for x in s):.2f} router_end_to_moe_start={med(x["router_end_to_moe_start_us"] for x in s):.2f}'
        if k=='STREAM':line+=f' MB={c["unique_bytes"]/1e6:.1f}'
        if k=='EMPTY_match' or k=='STREAM':
            first=[min(r[0] for r in x['stamps'] if r[0]) for x in s];line+=f' launch={med((f-x["markers"][0])/1000 for f,x in zip(first,s)):.2f}'
        out.append(line)
    names='route,zero,W13,W2ready,W2rest,fence,barrier,combine,footer'
    for k,rows in result['account'].items():
        g=lambda key:med(x[key] for x in rows)
        out.append(f'ACCOUNT {k} event={g("event_us"):.2f} marker_span={g("marker_span_us"):.2f} launch={g("launch_us"):.2f} CTAspan={g("cta_span_us"):.2f} drain={g("drain_us"):.2f} entry_spread={g("entry_spread_us"):.2f} lastW13end={g("last_W13_end_us"):.2f} afterLastW2={g("after_last_W2_us"):.2f} issue={g("first_copy_issue_us"):.2f} land={g("first_copy_land_us"):.2f} W2wait_med/max={g("w2_wait_med_us"):.2f}/{g("w2_wait_max_us"):.2f}'
            +(f' cupti_node={g("node_us"):.2f}' if rows[0].get('node_us') else ''))
        sk=[x['skew'] for x in rows];sm=[med(v[i] for v in sk) for i in range(13)]
        out.append(f'SKEW {k} W13end p50/p90/max={sm[0]:.2f}/{sm[1]:.2f}/{sm[2]:.2f} W2end={sm[3]:.2f}/{sm[4]:.2f}/{sm[5]:.2f} barrier_exit={sm[6]:.2f}/{sm[7]:.2f}/{sm[8]:.2f} units13={sm[9]:.0f}-{sm[10]:.0f} units2={sm[11]:.0f}-{sm[12]:.0f}')
        out.append(f'CRITICAL {k} {names}='+','.join(f'{med(x["phases"][i] for x in rows):.2f}' for i in range(9)))
        t={key:med(x['tile'][key] for x in rows) for key in rows[0]['tile']}
        out.append(f'TILE {k} per13/per2={t["per13"]:.2f}/{t["per2"]:.2f} t0_wait13/wait2/bar={t["wait13_t0"]:.2f}/{t["wait2_t0"]:.2f}/{t["bar_t0"]:.2f} w7_wait/bar={t["wait_w7"]:.2f}/{t["bar_w7"]:.2f}'
            f' pub_fence={t["fence"]:.2f}(max {t["fence_max"]:.2f}) pub_bar={t["pubbar"]:.2f}(max {t["pubbar_max"]:.2f}) pub_after_own13={t["pub_after_own13"]:.2f} last_pub_after_last13={t["last_pub_after_last13"]:.2f}')
    return out
