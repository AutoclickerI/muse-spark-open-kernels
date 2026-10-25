DC_SEEDS=tuple(5101199+104729*i for i in range(32))
DC_ROUNDS=10


def dc_activity():
    tag=hashlib.sha256(DC_ACTIVITY.encode()).hexdigest()[:12]
    return load_inline(name='bd_dag_'+tag,cpp_sources=DC_ACTIVITY,
        functions=['profile_begin','profile_end','capture_nodes','capture_dag','exec_id'],
        no_implicit_headers=True,with_pytorch_error_handling=False,
        extra_include_paths=['/usr/local/cuda/include'],extra_cflags=['-O2','-std=c++20'],
        extra_ldflags=['-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcupti','-lcudart'],verbose=False)


def dc_profile(graph,mapping,act):
    act.profile_begin()
    for _ in range(5):lab_cold_l2();graph.replay()
    torch.cuda.synchronize();records=act.profile_end();by={}
    for name,r in records:
        if r[4] in mapping:by.setdefault(r[4],[]).append((name,r))
    assert set(by)==set(mapping) and all(len(v)==5 for v in by.values())
    for v in by.values():v.sort(key=lambda a:a[1][0])
    traces=[[v[i] for v in by.values()] for i in range(5)]
    spans=[max(r[1] for _,r in tr)-min(r[0] for _,r in tr) for tr in traces]
    tr=traces[sorted(range(5),key=lambda i:spans[i])[2]];start=min(r[0] for _,r in tr)
    return dict(span_samples_us=[v/1000 for v in spans],nodes=[dict(node=r[4],stage=mapping[r[4]],name=n,stream=r[13],start_ns=r[0]-start,end_ns=r[1]-start,grid=list(r[7:10]),block=r[10],shared=r[11],registers=r[12]) for n,r in sorted(tr,key=lambda a:a[1][0])])


def dc_states(base,state):
    ns={};st={};reused={}
    for name,source in DC_SOURCES.items():
        v=pa_namespace(source,'d1_'+name);pa_share(base,v,source);reused[name]=pa_share_jit(base,v)
        if name!='D':v['GP_ROWS']=()
        ns[name]=v;st[name]=pa_state(state,[])
        for key,plan in st[name]['best7_plans'].items():plan['gp_key']=key
        if name!='D':st[name]['gp_roles']=('b1','route','b2')
    # Identical native sources deliberately have one module/actual kernel handle.
    for name in ns:
        if name in ('c4','D'):ns[name]['_GP_MODULE']=ns['weights']['gp_module']()
        else:ns[name]['gp_module']()
    arms=dict(A=(base,state),A2=(base,state),D=(ns['D'],st['D']))
    for label,name in [('W','weights'),('L','late_prew'),('H','hints'),('S','stream'),('C','c4')]:arms[label]=(ns[name],st[name])
    for role,label in [('b1','W1'),('route','WR'),('b2','W2')]:
        value=pa_state(st['weights'],[]);value['gp_roles']=(role,);arms[label]=(ns['weights'],value)
    return arms,reused


def dc_ready(base,stage_io,act):
    result={}
    for role in ('b1','route','b2'):
        args,out=stage_io[role]
        fn=base['_best7_route'] if role=='route' else base['_best7_boundary']
        graph,got=ab_graph(lambda args=args,fn=fn:fn(*args))
        ev=ab_time({'ready':(graph,got)},10)['ready'];act.profile_begin()
        for _ in range(5):lab_cold_l2();graph.replay()
        torch.cuda.synchronize();records=act.profile_end();rows=[(n,r) for n,r in records if ('route' if role=='route' else 'boundary') in n]
        assert len(rows)==5
        result[role]=dict(event_us=ev,kernel_us=[(r[1]-r[0])/1000 for _,r in rows],kernel_name=rows[0][0])
    return result


def dc_pack(result):
    # Every seed is retained. A is absolute ns; other times are signed deltas.
    samples=[];names=[];metadata=[]
    for case in result['cases']:
        rows=case.pop('samples');columns=list(rows[0]);case['columns']=columns;case['sample_rows']=len(rows);case['sample_offset']=len(samples)
        for row in rows:
            a=round(row['A']*1000)
            for col in columns:
                v=round(row[col]*1000);d=v-a;samples.append(v if col=='A' else (d<<1)^(d>>31))
        for arm,p in case['profiles'].items():
            origin=min(r[0] for r in case['dag'][arm]['nodes'] if r[0]>=0);case['dag'][arm]['id_origin']=origin
            for row in case['dag'][arm]['nodes']:
                if row[0]>=0:row[0]-=origin
            packed=[];last=0
            for node in p['nodes']:
                node=dict(node);name=node.pop('name')
                if name not in names:names.append(name)
                ident=node.pop('node');stream=node.pop('stream');start=node.pop('start_ns');end=node.pop('end_ns');node['name']=names.index(name)
                if node not in metadata:metadata.append(node)
                packed.append([ident-origin,metadata.index(node),stream,start-last,end-start]);last=start
            p['nodes']=packed
    result.update(version=1,node_names=names,node_metadata=metadata,sample_encoding='A absolute ns;others zigzag delta ns',binary_count=len(samples))
    def rounded(x):
        if isinstance(x,float):return round(x,4)
        if isinstance(x,dict):return {k:rounded(v) for k,v in x.items()}
        if isinstance(x,list):return [rounded(v) for v in x]
        return x
    header=json.dumps(rounded(result),separators=(',',':')).encode();raw=struct.pack('<I',len(header))+header+struct.pack('<'+'I'*len(samples),*samples)
    return base64.b85encode(lzma.compress(raw,preset=9)).decode()


def lab(deadline,weights,state=None):
    started=time.monotonic();assert weights is None and state is None
    weights=generate_weights(5353,device='cuda',router_seed=197731)
    base=pa_namespace(PA_BASE,'d1_base');state=base['prepare'](weights)
    if torch.cuda.get_device_capability()!=(9,0):base['_ap_fa3_scheduler_metadata']=lambda *a,**kw:None
    arms,reused=dc_states(base,state);act=dc_activity();cases=[]
    for window in (2048,0):
        assert time.monotonic()<deadline-20,'insufficient budget before case'
        mk=lambda seed:base['_ap_generate_input'](batch=1,ctx=80000,layer='sliding' if window else 'full',q_len=1,spread='fixed',dist='soft',page=1,seed=seed,device='cuda')
        data=mk(DC_SEEDS[0]);graphs={};saved={};dag={};profiles={};gates={};quality={};samples=[]
        for name,(ns,st) in arms.items():ns['kernel_fn'](data,st)
        torch.cuda.synchronize()
        for name,(ns,st) in arms.items():
            if name in ('A','W','L','H','S','C'):
                graph,out,mapping,io,edges=bd_capture(ns,st,data,act);graphs[name]=(graph,out);saved[name]=(mapping,io);dag[name]=edges
            else:graphs[name]=ab_graph(lambda ns=ns,st=st:ns['kernel_fn'](data,st))
        for seed in DC_SEEDS:
            assert time.monotonic()<deadline-10,'insufficient budget during seeds'
            if seed!=DC_SEEDS[0]:ab_copy(data,mk(seed))
            for graph,_ in graphs.values():graph.replay()
            torch.cuda.synchronize()
            for name,(_,out) in graphs.items():
                if name=='A':continue
                q=pa_quality(out,graphs['A'][1]);old=quality.setdefault(name,dict(id_rows=0,l2=0.))
                old['id_rows']=max(old['id_rows'],q['id_rows']);old['l2']=max(old['l2'],q['l2'])
                if why:=ab_gate(out,graphs['A'][1]):gates.setdefault(name,why)
                if q['id_rows']:gates.setdefault(name,'route membership changed; requires official margin review')
            values=ab_time(graphs,DC_ROUNDS);samples.append({n:statistics.mean(v) for n,v in values.items()})
        unprofiled={n:statistics.median(v) for n,v in ab_time({n:graphs[n] for n in saved},5).items()}
        for name,(mapping,_) in saved.items():profiles[name]=dc_profile(graphs[name][0],mapping,act)
        ready=dc_ready(base,saved['A'][1],act)
        cases.append(dict(case='d1S' if window else 'd1F',samples=samples,gates=gates,quality=quality,profiles=profiles,dag=dag,ready=ready,unprofiled_event_us=unprofiled,plan=state['best7_plans'][(1,1,window)]))
        del graphs,data,saved;gc.collect();torch.cuda.empty_cache()
    result=dict(base_sha256=hashlib.sha256(PA_BASE.encode()).hexdigest(),source_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in DC_SOURCES.items()},cases=cases,seeds=list(DC_SEEDS),rounds=DC_ROUNDS,reused_jit=reused)
    blob=dc_pack(result)
    lines=['META S14 d1 glue;32seeds*10coldREAD;unchanged JIT/native handles shared;A2=A:A',
        'META W=allpreload,W1/WR/W2=roles,L=lateprew,H=hints,S=b1stream,C=C4,D=priorfull;CUPTI/DAG+same-handle ready',
        'D1_CHAIN_XZ85 '+blob,'END cases=2 elapsed='+str(round(time.monotonic()-started,1))]
    assert sum(len(s)+1 for s in lines)<9000,'archive exceeds 9KB receipt'
    return lines
