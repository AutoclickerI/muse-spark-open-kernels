T15_CASES=tuple((b,q,w) for q in (1,8) for b in (1,16,32) for w in (2048,0))
T15_SEEDS=(7101293,7206022,7310751)
T15_REPS=3


def t15_activity():
    tag=hashlib.sha256(T15_ACTIVITY.encode()).hexdigest()[:12]
    return load_inline(name='s15_timeline_'+tag,cpp_sources=T15_ACTIVITY,
        functions=['profile_begin','profile_end','capture_nodes','capture_dag','exec_id'],no_implicit_headers=True,with_pytorch_error_handling=False,
        extra_include_paths=['/usr/local/cuda/include'],extra_cflags=['-O2','-std=c++20'],
        extra_ldflags=['-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcupti','-lcudart'],verbose=False)


def t15_namespace():
    original=kernel_fn.__globals__;ns=dict(original)
    for name,fn in original.items():
        if isinstance(fn,types.FunctionType) and fn.__globals__ is original:
            copied=types.FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__,fn.__closure__);copied.__kwdefaults__=fn.__kwdefaults__;copied.__dict__.update(fn.__dict__);ns[name]=copied
    return ns


def t15_capture(fn,state,data,act,ns=None):
    roles={};io={};originals={}
    def wrap(name,label):
        original=ns[name];originals[name]=original
        def call(*args,**kwargs):
            before=set(act.capture_nodes(torch.cuda.current_stream().cuda_stream));out=original(*args,**kwargs);after=set(act.capture_nodes(torch.cuda.current_stream().cuda_stream));role=label(args)
            assert not(set(roles)&(after-before)),'nested stage wrapper'
            roles.update({i:role for i in after-before});io[role]=(args,out);return out
        ns[name]=call
    if ns is not None:
        wrap('_best7_project',lambda a:a[1]);wrap('_best7_attention',lambda a:'attn');wrap('_gate_split_apply',lambda a:'gate')
        wrap('_best7_boundary',lambda a:a[4]);wrap('_best7_route',lambda a:'route');wrap('_best7_expert',lambda a:'moe')
    try:
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=fn(data,state);dag=act.capture_dag(torch.cuda.current_stream().cuda_stream)
    finally:
        if ns is not None:ns.update(originals)
    eid=act.exec_id(graph.raw_cuda_graph_exec());rekey=lambda i:(eid<<32)|(i&0xffffffff) if i>=0 else i
    rows=[[rekey(row[0]),row[1]] for row in dag[0]];roles={rekey(i):v for i,v in roles.items()}
    for row in rows:
        if row[1] in (0,1,2) and row[0] not in roles:roles[row[0]]='setup'
    return graph,out,roles,io,dict(nodes=rows,edges=dag[1])


def t15_profile(graph,roles,dag,act,window):
    act.profile_begin();events=[]
    for _ in range(T15_REPS):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        lab_cold_l2();a.record();graph.replay();b.record();events.append((a,b))
    torch.cuda.synchronize();records=act.profile_end();by={}
    for name,r in records:
        if r[4] in roles:by.setdefault(r[4],[]).append((name,r))
    assert set(by)==set(roles) and all(len(v)==T15_REPS for v in by.values()),'missing/duplicate CUPTI graph activity'
    for v in by.values():v.sort(key=lambda item:item[1][0])
    traces=[];accounts=[]
    for i in range(T15_REPS):
        trace=[];origin=min(v[i][1][0] for v in by.values())
        for values in by.values():
            name,r=values[i];role=roles[r[4]]
            if role=='attn':role='attention_sliding' if window else 'attention_full'
            trace.append(dict(node=r[4],role=role,name=name,stream=r[13],start_ns=r[0]-origin,end_ns=r[1]-origin,grid=list(r[7:10]),block=r[10],shared=r[11],registers=r[12]))
        trace.sort(key=lambda n:n['start_ns']);traces.append(trace);accounts.append(chain_account(trace,dag))
    chosen=sorted(range(T15_REPS),key=lambda i:accounts[i]['wall_us'])[T15_REPS//2]
    return dict(trace=traces[chosen],account=accounts[chosen],sample_wall_us=[a['wall_us'] for a in accounts],event_us=[a.elapsed_time(b)*1000 for a,b in events],max_accounting_error=max(a['accounting_error_fraction'] for a in accounts),all_stamps_monotone=all(a['monotone'] for a in accounts))


def t15_signature(trace,dag):
    # Names, launch geometry/resources, streams and dependency types. Graph IDs
    # change on instantiation; normalize them without changing topology.
    by={n['node']:n for n in trace};streams={}
    rows=[]
    for ident,typ in dag['nodes']:
        n=by.get(ident)
        if n:
            if n['stream'] not in streams:streams[n['stream']]=len(streams)
            rows.append([typ,n['name'],n['grid'],n['block'],n['shared'],n['registers'],streams[n['stream']]])
        else:rows.append([typ])
    return dict(nodes=rows,edges=dag['edges'])


def t15_busiest(ns,data,state):
    # Choose the busiest public expert shard from S15's actual front, matching
    # the judge's busiest-shard intent; actual counts are retained per fixture.
    out=ns['kernel_fn'](data,state);torch.cuda.synchronize();ids=out[2]
    shard=int(torch.bincount((ids//32).flatten(),minlength=8).argmax())
    data['local_start'].fill_(shard*32)
    return shard*32


def t15_pack(result):
    names=[];metadata=[];dags=[];binary=[]
    result['max_accounting_error']=max(s['max_accounting_error'] for c in result['cases'] for s in c['seeds'])
    result['all_stamps_monotone']=all(s['all_stamps_monotone'] for c in result['cases'] for s in c['seeds'])
    stages=sorted({stage for cell in result['cases'] for seed in cell['seeds'] for stage in seed['account']['critical_us']})
    zig=lambda x:(x<<1)^(x>>63)
    for cell in result['cases']:
        trace=cell['trace'];dag=cell['dag'];origin=min(n[0] for n in dag['nodes'] if n[0]>=0)
        cell['projection_storage']={k:v.get('weights_bytes') for k,v in cell['seeds'][0]['details'].items() if isinstance(v,dict) and 'encoding' in v}
        first=cell['seeds'][0];dynamic=('attention_sliding','attention_full','moe_small','moe_light','moe_heavy','moe_combine')
        cell['work_static']={k:[v[f] for f in ('read_bytes','write_bytes','bf16_flops','fp8_flops')] for k,v in first['work'].items() if k not in dynamic}
        attn_role=next(k for k in first['work'] if k.startswith('attention'))
        aw=first['work'][attn_role];ad=first['details'][attn_role]
        cell['attention_recipe']=[attn_role,aw['read_bytes']-ad['unique_keys']*260,aw['write_bytes'],bool(aw['fp8_flops'])]
        cell['control_event_us']={key:round(statistics.mean(v for seed in cell['seeds'] for v in seed[key]),2) for key in ('profile_event_us','plain_event_us','mapped_event_us')}
        for seed in cell['seeds']:
            for key in cell['control_event_us']:seed.pop(key)
            seed.pop('work');seed.pop('max_accounting_error');seed.pop('all_stamps_monotone')
            details=seed.pop('details')
            attention=next(v for k,v in details.items() if k.startswith('attention'))
            seed['fixture']=[details['expert_counts'],attention['unique_keys'],attention['causal_query_keys']]
            account=seed.pop('account');wall=round(account['wall_us']*1000)
            seed['critical_offset']=len(binary)
            binary.extend(round(account['critical_us'].get(stage,0)*1000) for stage in stages)
            seed['joins_offset']=len(binary);seed['joins_count']=len(account['joins'])
            for join in account['joins']:
                binary.extend(zig(join[k]-origin if join[k]>=0 else join[k]) for k in ('node','winner','runner_up'))
                binary.append(round(join['slack_us']*1000))
            for key in ('profile_wall_samples_us',):
                values=seed.pop(key)
                if key!='profile_wall_samples_us':values=[statistics.mean(values)]
                seed[key]=[len(binary),len(values)]
                # Event resolution is .01 us; wall samples remain integer ns.
                scale=1000 if key=='profile_wall_samples_us' else 100
                center=round(wall*scale/1000)
                binary.extend(zig(round(v*scale)-center) for v in values)
        template=dict(nodes=[[i-origin if i>=0 else i,t] for i,t in dag['nodes']],edges=dag['edges'])
        if template not in dags:dags.append(template)
        cell['dag']=dict(template=dags.index(template),id_origin=origin);cell['trace_offset']=len(binary);cell['trace_count']=len(trace);last=0
        for node in trace:
            n=dict(node);ident=n.pop('node');stream=n.pop('stream');start=n.pop('start_ns');end=n.pop('end_ns');name=n.pop('name')
            if name not in names:names.append(name)
            n['role']=stage_name(n['role'],name);n['name']=names.index(name)
            if n not in metadata:metadata.append(n)
            binary.extend((ident-origin,metadata.index(n),stream,start-last,end-start));last=start
        del cell['trace']
    result.update(version=1,stages=stages,name_legend_sha256_prefix=hashlib.sha256(json.dumps(names).encode()).hexdigest()[:16],names=[n.replace('void ','').replace('(anonymous namespace)::','').split('(')[0][:16]+'@'+str(i) for i,n in enumerate(names)],node_metadata=metadata,dag_templates=dags,binary_count=len(binary))
    def rounded(x):
        if isinstance(x,float):return round(x,4)
        if isinstance(x,dict):return {k:rounded(v) for k,v in x.items()}
        if isinstance(x,list):return [rounded(v) for v in x]
        return x
    text=json.dumps(rounded(result),separators=(',',':')).encode();raw=bytearray(struct.pack('<I',len(text))+text)
    for v in binary:
        assert v>=0
        while v>=128:raw.append((v&127)|128);v>>=7
        raw.append(v)
    return base64.b85encode(lzma.compress(raw,preset=9)).decode()


def lab(deadline,weights,state=None):
    assert weights is not None and state is not None,'use S15 as the decode host, ordinary prepare phase'
    started=time.monotonic();ns=t15_namespace();act=t15_activity();cases=[];shapes=[]
    empty=torch.cuda.CUDAGraph()
    with torch.cuda.graph(empty):pass
    empty_us=statistics.median(ab_time({'empty':(empty,None)},8)['empty'])
    for b,q,window in T15_CASES:
        assert time.monotonic()<deadline-8,'timeline budget exhausted before shape'
        mk=lambda seed:ns['_ap_generate_input'](batch=b,ctx=80000,layer='sliding' if window else 'full',q_len=q,spread='fixed' if b==1 else 'agentic',dist='soft',page=1,seed=seed,device='cuda')
        data=mk(T15_SEEDS[0]);start=t15_busiest(ns,data,state)
        # Warm original and cloned schedules before any activity recording.
        kernel_fn(data,state);ns['kernel_fn'](data,state);torch.cuda.synchronize()
        plain,pout,proles,_,pdag=t15_capture(kernel_fn,state,data,act)
        mapped,out,roles,io,dag=t15_capture(ns['kernel_fn'],state,data,act,ns)
        pprofile=t15_profile(plain,proles,pdag,act,window)
        seeds=[];alltraces=[];gates=[];signature=None
        for seed in T15_SEEDS:
            assert time.monotonic()<deadline-5,'timeline budget exhausted during seeds'
            if seed!=T15_SEEDS[0]:
                fresh=mk(seed);start=t15_busiest(ns,fresh,state);ab_copy(data,fresh);del fresh
            plain.replay();mapped.replay();torch.cuda.synchronize()
            why=ab_gate(out,pout)
            if why:gates.append(dict(seed=seed,why=why))
            times=ab_time({'plain':(plain,pout),'mapped':(mapped,out)},4)
            profile=t15_profile(mapped,roles,dag,act,window)
            if signature is None:
                signature=t15_signature(profile['trace'],dag)
                assert signature==t15_signature(pprofile['trace'],pdag),'capture instrumentation changed the kernel sequence, resources, streams or DAG'
            work,details=t15_work(data,state,io,ns)
            seeds.append(dict(seed=seed,local_start=start,account=profile['account'],work=work,details=details,profile_wall_samples_us=profile['sample_wall_us'],profile_event_us=profile['event_us'],plain_event_us=times['plain'],mapped_event_us=times['mapped'],all_stamps_monotone=profile['all_stamps_monotone'],max_accounting_error=profile['max_accounting_error']))
            alltraces.append(profile['trace'])
        assert not gates,('mapped versus untouched S15 gate',gates)
        representative=sorted(range(len(seeds)),key=lambda i:seeds[i]['account']['wall_us'])[len(seeds)//2]
        cases.append(dict(case=('d' if q==1 else 'v')+str(b)+('S' if window else 'F'),weight=46 if window else 16,seeds=seeds,trace=alltraces[representative],trace_seed=T15_SEEDS[representative],dag=dag,schedule_identical=True,gates=gates))
        del plain,mapped,io,data,out,pout,alltraces;gc.collect();torch.cuda.empty_cache()
    result=dict(s15_sha256=T15_SOURCE_SHA,cases=cases,profile_reps=T15_REPS,seeds=list(T15_SEEDS),empty_event_us=empty_us,roof_model=dict(stream_bps=STREAM_BPS,stream_intercept_us=STREAM_INTERCEPT_US,bf16_flops=BF16_FLOPS,fp8_flops=FP8_FLOPS,hbm_bps=HBM_BPS),elapsed_s=time.monotonic()-started)
    blob=t15_pack(result);lines=['META S15 60254883;CUPTI all streams;device kernels unchanged',
        'META 12shapes*3seeds*3profiles;DAG critical chain;logical byte/FLOP roofs',
        'S15_TIMELINE_XZ85 '+blob,'END elapsed='+str(round(time.monotonic()-started,1))]
    assert sum(len(s)+1 for s in lines)<9000,'timeline archive exceeds receipt cap'
    return lines
