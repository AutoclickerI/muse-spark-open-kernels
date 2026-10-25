BD_SHARED={}
BD_REPS=5


def pa_namespace(source,name):
    ns=BD_NAMESPACE_ORIGINAL(source,name)
    if name=='follow_A':
        original=ns['prepare']
        def prepare(weights):
            state=original(weights);BD_SHARED.update(base=ns,state=state,weights=weights)
            return state
        ns['prepare']=prepare
    return ns


def bd_activity():
    tag=hashlib.sha256(BD_ACTIVITY.encode()).hexdigest()[:12]
    return load_inline(name='bd_dag_'+tag,cpp_sources=BD_ACTIVITY,
        functions=['profile_begin','profile_end','capture_nodes','capture_dag','exec_id'],
        no_implicit_headers=True,with_pytorch_error_handling=False,
        extra_include_paths=['/usr/local/cuda/include'],extra_cflags=['-O2','-std=c++20'],
        extra_ldflags=['-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcupti','-lcudart'],verbose=False)


def bd_capture(ns,st,data,act):
    mapping={};saved={};originals={}
    def wrap(name,label):
        original=ns[name];originals[name]=original
        def call(*args,**kw):
            before=set(act.capture_nodes(torch.cuda.current_stream().cuda_stream))
            out=original(*args,**kw)
            after=set(act.capture_nodes(torch.cuda.current_stream().cuda_stream))
            stage=label(args);ids=after-before
            assert not(set(mapping)&ids),('nested stage',stage)
            mapping.update({i:stage for i in ids});saved[stage]=(args,out)
            return out
        ns[name]=call
    wrap('_best7_project',lambda a:a[1]);wrap('_best7_attention',lambda a:'attn')
    wrap('_gate_split_apply',lambda a:'gate');wrap('_best7_boundary',lambda a:a[4])
    wrap('_best7_route',lambda a:'route');wrap('_best7_expert',lambda a:'moe')
    try:
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out=ns['kernel_fn'](data,st)
            dag=act.capture_dag(torch.cuda.current_stream().cuda_stream)
    finally:ns.update(originals)
    for row in dag[0]:
        if row[1]==0 and row[0] not in mapping:mapping[row[0]]='init'
    eid=act.exec_id(graph.raw_cuda_graph_exec())
    rekey=lambda i:(eid<<32)|(i&0xffffffff) if i>=0 else i
    mapping={rekey(i):s for i,s in mapping.items()}
    nodes=[[rekey(row[0]),*row[1:]] for row in dag[0]]
    return graph,out,mapping,saved,dict(nodes=nodes,edges=dag[1])


def bd_profile(graph,act,mapping):
    act.profile_begin()
    try:
        for _ in range(BD_REPS):lab_cold_l2();graph.replay()
        torch.cuda.synchronize()
        records=act.profile_end()
    except BaseException:
        try:act.profile_end()
        except BaseException:pass
        raise
    by={}
    for name,r in records:
        if r[4] in mapping:by.setdefault(r[4],[]).append((name,r))
    assert set(by)==set(mapping),(set(mapping)-set(by))
    assert all(len(v)==BD_REPS for v in by.values())
    for v in by.values():v.sort(key=lambda item:item[1][0])
    traces=[[v[i] for v in by.values()] for i in range(BD_REPS)]
    spans=[(max(r[1] for _,r in tr)-min(r[0] for _,r in tr))/1000 for tr in traces]
    chosen=sorted(range(BD_REPS),key=lambda i:spans[i])[BD_REPS//2];trace=traces[chosen]
    origin=min(r[0] for _,r in trace);nodes=[]
    for name,r in sorted(trace,key=lambda item:item[1][0]):
        nodes.append(dict(node=r[4],stage=mapping[r[4]],name=name,stream=r[13],start=(r[0]-origin)/1000,end=(r[1]-origin)/1000,
            grid=list(r[7:10]),block=r[10],shared=r[11],registers=r[12]))
    stages={s:dict(start=min(n['start'] for n in nodes if n['stage']==s),end=max(n['end'] for n in nodes if n['stage']==s)) for s in set(mapping.values())}
    previous=0.;increments={}
    for s in ('init','qkv3','qkv_gate','attn','gate','o','b1','pool_router','route','moe','unpooler','b2'):
        if s in stages:
            end=stages[s]['end'];increments[s]=max(0.,end-previous);previous=max(previous,end)
    boundaries={}
    for role,producer in (('b1','o'),('b2','unpooler')):
        b,p=stages[role],stages[producer]
        boundaries[role]=dict(residence=b['end']-b['start'],increment=increments[role],
            before_producer_end=max(0.,min(b['end'],p['end'])-b['start']),
            after_producer_end=max(0.,b['end']-max(b['start'],p['end'])),
            start_minus_producer_end=b['start']-p['end'])
    return dict(wall=spans[chosen],span_samples=spans,nodes=nodes,stages=stages,increments=increments,boundaries=boundaries)


def bd_states(base,state):
    namespaces={};states={};private={'_SMALLM_MODULE'}
    for name,source in BD_SOURCES.items():
        ns=BD_NAMESPACE_ORIGINAL(source,'bd_'+name)
        pa_share(base,ns,source,private if name in ('R','RP') else ())
        pa_share_jit(base,ns);namespaces[name]=ns
        st=pa_state(state,[])
        if name in ('C','S'):
            for key,plan in st['best7_plans'].items():
                if key!='other' and key[0] in (16,32) and key[1]==1:
                    if name=='C':plan['geometry'].update(b1=('cl',4,256,1),b2=('cl',4,256,1))
                    else:plan['bd_serial']=True
        states[name]=st
    namespaces['R']['smallm_module']()
    namespaces['RP']['_SMALLM_MODULE']=namespaces['R']['_SMALLM_MODULE']
    namespaces['RP']['_bd_unpool_kernel']=namespaces['P']['_bd_unpool_kernel']
    namespaces['S']['bd_serial_module']()
    return namespaces,states


def bd_run(deadline):
    base,state=BD_SHARED['base'],BD_SHARED['state']
    ns,states=bd_states(base,state);ns.update(A=base,A2=base);states.update(A=state,A2=state)
    act=bd_activity();cases=[];seeds=tuple(4101119+104729*i for i in range(16))
    for b in (16,32):
        for window in (2048,0):
            assert time.monotonic()<deadline-8,'boundary lab budget exhausted'
            label='d'+str(b)+('S' if window else 'F')
            mk=lambda seed:base['_ap_generate_input'](batch=b,ctx=80000,layer='sliding' if window else 'full',q_len=1,
                spread='agentic',dist='soft',page=1,seed=seed,device='cuda')
            data=mk(seeds[0]);graphs={};gates={};quality={};profiles={};dag={};saved={}
            # Warm every selected arm and native path before graph capture.
            for name in states:
                ns[name]['kernel_fn'](data,states[name])
            torch.cuda.synchronize()
            for name in states:
                if name in ('A','RP','S','C'):
                    graph,out,mapping,stage_io,edges=bd_capture(ns[name],states[name],data,act)
                    graphs[name]=(graph,out);dag[name]=edges;saved[name]=(mapping,stage_io)
                else:graphs[name]=ab_graph(lambda name=name:ns[name]['kernel_fn'](data,states[name]))
            samples=[]
            for seed in seeds:
                if seed!=seeds[0]:ab_copy(data,mk(seed))
                for graph,_ in graphs.values():graph.replay()
                torch.cuda.synchronize()
                for name,(_,out) in graphs.items():
                    if name!='A':
                        if why:=ab_gate(out,graphs['A'][1]):gates.setdefault(name,why)
                        q=pa_quality(out,graphs['A'][1]);old=quality.setdefault(name,dict(id_rows=0,l2=0.))
                        old['id_rows']=max(old['id_rows'],q['id_rows']);old['l2']=max(old['l2'],q['l2'])
                values=ab_time(graphs,8);samples.append({n:statistics.mean(v) for n,v in values.items()})
            off={n:statistics.median(tl) for n,tl in ab_time({n:graphs[n] for n in saved},5).items()}
            for name,(mapping,_) in saved.items():profiles[name]=bd_profile(graphs[name][0],act,mapping)
            # Same actual kernel handle, same ready buffers, no producing grid in these graphs.
            ready={};footprints={}
            for role in ('b1','b2'):
                args,outbuf=saved['A'][1][role]
                branch,stream,st,plan,phase,*_=args;p,split=branch
                weights=('alpha2','beta2','next_pre_attn_w','post_ffn_w') if phase=='b2' else ('alpha1','beta1','pre_ffn_w')
                reads=p.numel()*p.element_size()+stream.numel()*stream.element_size()+sum(st[k].numel()*st[k].element_size() for k in weights)
                writes=sum(t.numel()*t.element_size() for t in outbuf)
                footprints[role]=dict(partial_shape=list(p.shape),partial_dtype=str(p.dtype),split=split,stream_shape=list(stream.shape),read_bytes=reads,write_bytes=writes,read_roof_us=2.36+reads/4.58e6,traffic_roof_us=2.36+(reads+writes)/4.58e6)
                graph,out=ab_graph(lambda args=args:base['_best7_boundary'](*args))
                ready[role]=dict(event_us=ab_time({'ready':(graph,out)},8)['ready'])
                ids=None
                # Single-kernel activity does not need stage capture queries.
                act.profile_begin()
                for _ in range(5):lab_cold_l2();graph.replay()
                torch.cuda.synchronize();records=act.profile_end()
                rows=[(n,r) for n,r in records if 'boundary' in n]
                assert len(rows)==5
                ready[role]['kernel_us']=[(r[1]-r[0])/1000 for _,r in rows]
            cfg=states['A']['best7_plans'][(b,1,window)]
            cases.append(dict(case=label,gates=gates,quality=quality,samples=samples,profiles=profiles,dag=dag,ready=ready,
                unprofiled_event_us=off,footprints=footprints,plan=cfg))
            del graphs,data,saved;gc.collect();torch.cuda.empty_cache()
    return cases


def bd_pack(confirmation,boundaries):
    # Preserve all per-seed times in binary nanoseconds. Keep every seed; timing resolution is 0.001 us, other floats 0.0001.
    binary=[]
    def pack_cells(cells):
        out=[]
        for cell in cells:
            row=dict(cell);samples=row.pop('samples');names=list(samples[0]);offset=len(binary)
            for sample in samples:
                baseline=round(sample['A']*1000)
                for n in names:
                    value=round(sample[n]*1000);delta=value-baseline
                    binary.append(value if n=='A' else (delta<<1)^(delta>>31))
            row['sample_columns']=names;row['sample_offset']=offset;row['sample_rows']=len(samples);out.append(row)
        return out
    confirmation=dict(confirmation);confirmation['cells']=pack_cells(confirmation['cells'])
    boundaries=pack_cells(boundaries)
    # Kernel identities/resources are shared across traces. Derived completion
    # frontiers are recomputed from the retained intervals by report.py.
    names=[];metadata=[]
    for case in boundaries:
        for arm,profile in case['profiles'].items():
            dag=case['dag'][arm];origin=min(r[0] for r in dag['nodes'] if r[0]>=0)
            dag['id_origin']=origin
            for row in dag['nodes']:
                if row[0]>=0:row[0]-=origin
            packed=[];previous_start=0
            for node in profile['nodes']:
                node=dict(node);name=node.pop('name')
                if name not in names:names.append(name)
                ident=node.pop('node');stream=node.pop('stream');start=node.pop('start');end=node.pop('end')
                node['name']=names.index(name)
                if node not in metadata:metadata.append(node)
                start_ns=round(start*1000);end_ns=round(end*1000)
                packed.append([ident-origin,metadata.index(node),stream,start_ns-previous_start,end_ns-start_ns]);previous_start=start_ns
            spans=profile['span_samples'];wall=profile['wall']
            profile.clear();profile.update(nodes=packed,span_samples=spans,wall=wall,interval_unit='0.001 us')
    header=dict(version=2,confirmation=confirmation,boundaries=boundaries,names=names,node_metadata=metadata,
        sample_unit='0.001 us',sample_encoding='A absolute;others zigzag delta from A',binary_dtype='little-endian uint32',binary_count=len(binary))
    def rounded(x):
        if isinstance(x,float):return round(x,4)
        if isinstance(x,dict):return {k:rounded(v) for k,v in x.items()}
        if isinstance(x,list):return [rounded(v) for v in x]
        return x
    text=json.dumps(rounded(header),separators=(',',':')).encode()
    raw=struct.pack('<I',len(text))+text+struct.pack('<'+'I'*len(binary),*binary)
    return base64.b85encode(lzma.compress(raw,preset=9)).decode()


def lab(deadline,weights,state=None):
    assert weights is None and state is None,'batch owns synthetic setup at import'
    started=time.monotonic()
    lines=BD_CONFIRM_RUN(deadline-120,None,None)
    archive=next(line.split(' ',1)[1] for line in lines if line.startswith('PDL_AUDIT_XZ85 '))
    confirmation=json.loads(lzma.decompress(base64.b85decode(archive)))
    assert len(confirmation['cells'])==12 and all(not c['gates'] and c['seeds']==16 for c in confirmation['cells'])
    boundaries=bd_run(deadline)
    blob=bd_pack(confirmation,boundaries)
    out=['META exact6cb428bc+S14 boundaries d16/d32;synthetic;DAG+CUPTI',
        'META R=MoErelease;P=unpoolhint;RP=R+P;C=C4;S=serial;16seeds*8coldREAD',
        'BD_BATCH_XZ85 '+blob,'END confirmation=12 boundary=4 elapsed='+str(round(time.monotonic()-started,1))]
    assert sum(len(line)+1 for line in out)<9000,'combined archive exceeds lab log cap'
    return out
