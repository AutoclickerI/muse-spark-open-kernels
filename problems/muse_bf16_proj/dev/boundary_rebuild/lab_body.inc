BR_SEEDS=tuple(6101277+104729*i for i in range(24))
BR_ROUNDS=8


def br_states(base,state):
    namespaces={};states={};reused={}
    for kind,source in BR_SOURCES.items():
        ns=pa_namespace(source,'br_'+kind);pa_share(base,ns,source);reused[kind]=pa_share_jit(base,ns);namespaces[kind]=ns
        st=pa_state(state,[])
        for key,plan in st['best7_plans'].items():
            if key!='other' and key[0] in (16,32) and key[1] in (1,8):ns['configure'](plan,kind,key[1]==1)
        states[kind]=st
    module=namespaces['exact']['br_module']()
    for ns in namespaces.values():ns['_BR_MODULE']=module
    arms=dict(A=(base,state),A2=(base,state))
    for label,kind in [('M','moment'),('E','exact'),('S','serial'),('R','route'),('C','chain'),('P','publish')]:arms[label]=(namespaces[kind],states[kind])
    for label,role in [('E1','b1'),('E2','b2')]:
        st=pa_state(states['exact'],[])
        for key,plan in st['best7_plans'].items():
            if key!='other' and key[0] in (16,32) and key[1] in (1,8):plan['br_roles']=(role,)
        arms[label]=(namespaces['exact'],st)
    return arms,reused


def br_ready(arms,saved):
    result={}
    for role in ('b1','b2','route'):
        result[role]={}
        for label in (('A','R') if role=='route' else ('A','M','E')):
            ns,st=arms[label];args=list(saved[role][0]);key=(args[3].get('br_key') or saved['key'])
            args[3]=st['best7_plans'][key]
            fn=ns['_best7_route'] if role=='route' else ns['_best7_boundary']
            graph,out=ab_graph(lambda fn=fn,args=args:fn(*args));values=ab_time({'ready':(graph,out)},10)['ready']
            result[role][label]=dict(median=statistics.median(values),min=min(values),max=max(values),n=len(values))
    return result


def br_pack(result):
    # Preserve all paired seeds as ratios (0.0001 resolution) plus baseline
    # latency (0.01 us). Keep full dependency DAGs; record critical role kernels
    # and stage envelopes for the remaining work. Verify P is identical to A.
    result['source_sha256']={k:v[:16] for k,v in result['source_sha256'].items()}
    result['source_hash_prefix_chars']=16
    result['reused_jit']={k:len(v) for k,v in result['reused_jit'].items()}
    for cell in result['cases']:
        cell.pop('plan')
        for choices in cell['ready'].values():
            for record in choices.values():record.pop('min');record.pop('max')
        if cell['case'].startswith('v'):del cell['profiles']['P'],cell['dag']['P']
        for arm,p in cell['profiles'].items():
            stages={n['node']:n['stage'] for n in p['nodes']}
            for node in cell['dag'][arm]['nodes']:node.append(stages.get(node[0],'join'))
            kept=('o','b1','pool_router','route','unpooler','b2')
            p['stage_envelopes']={s:dict(start_ns=min(n['start_ns'] for n in p['nodes'] if n['stage']==s),end_ns=max(n['end_ns'] for n in p['nodes'] if n['stage']==s)) for s in sorted({n['stage'] for n in p['nodes']}) if s not in kept}
            p['nodes']=[n for n in p['nodes'] if n['stage'] in kept]
            p.pop('span_samples_us')
    binary=[];names=[];metadata=[];dag_templates=[]
    for cell in result['cases']:
        samples=cell.pop('samples');columns=list(samples[0]);cell.update(columns=columns,sample_offset=len(binary),sample_rows=len(samples))
        for sample in samples:
            a=round(sample['A']*100)
            for n in columns:
                delta=round((sample[n]/sample['A']-1)*10000);binary.append(a if n=='A' else (delta<<1)^(delta>>31))
        for arm,p in cell['profiles'].items():
            dag=cell['dag'][arm];origin=min(row[0] for row in dag['nodes'] if row[0]>=0);dag['id_origin']=origin
            for row in dag['nodes']:
                if row[0]>=0:row[0]-=origin
            packed=[];last=0
            for node in p['nodes']:
                node=dict(node);name=node.pop('name')
                if name not in names:names.append(name)
                ident=node.pop('node');stream=node.pop('stream');start=node.pop('start_ns');end=node.pop('end_ns');node['name']=names.index(name)
                if node not in metadata:metadata.append(node)
                packed.append([ident-origin,metadata.index(node),stream,start-last,end-start]);last=start
            p['node_offset']=len(binary);p['node_count']=len(packed)
            for row in packed:binary.extend(row)
            del p['nodes']
            template=dict(nodes=dag['nodes'],edges=dag['edges'])
            if template not in dag_templates:dag_templates.append(template)
            cell['dag'][arm]=dict(id_origin=origin,template=dag_templates.index(template))
    result.update(version=1,dag_templates=dag_templates,node_names=names,node_metadata=metadata,binary_count=len(binary),sample_encoding='A 0.01us;others zigzag ratio delta 0.0001')
    def rounded(x):
        if isinstance(x,float):return round(x,4)
        if isinstance(x,dict):return {k:rounded(v) for k,v in x.items()}
        if isinstance(x,list):return [rounded(v) for v in x]
        return x
    header=json.dumps(rounded(result),separators=(',',':')).encode();raw=bytearray(struct.pack('<I',len(header))+header)
    for v in binary:
        while v>=128:raw.append((v&127)|128);v>>=7
        raw.append(v)
    return base64.b85encode(lzma.compress(raw,filters=[dict(id=lzma.FILTER_LZMA2,preset=9,pb=0)])).decode()


def lab(deadline,weights,state=None):
    started=time.monotonic();assert weights is None and state is None
    weights=generate_weights(5353,device='cuda',router_seed=197731);base=pa_namespace(PA_BASE,'br_base');state=base['prepare'](weights)
    if torch.cuda.get_device_capability()!=(9,0):base['_ap_fa3_scheduler_metadata']=lambda *a,**kw:None
    arms,reused=br_states(base,state);act=dc_activity();cases=[]
    for b,q in ((16,1),(32,1),(16,8),(32,8)):
        for window in (2048,0):
            assert time.monotonic()<deadline-10,'budget before shape';key=(b,q,window)
            mk=lambda seed:base['_ap_generate_input'](batch=b,ctx=80000,layer='sliding' if window else 'full',q_len=q,spread='agentic',dist='soft',page=1,seed=seed,device='cuda')
            data=mk(BR_SEEDS[0]);graphs={};saved={};profiles={};dag={};samples=[];gates={};quality={}
            for name,(ns,st) in arms.items():ns['kernel_fn'](data,st)
            torch.cuda.synchronize()
            for name,(ns,st) in arms.items():
                if name in ('A','P'):
                    graph,out,mapping,io,edges=bd_capture(ns,st,data,act);graphs[name]=(graph,out);saved[name]=(mapping,io);dag[name]=edges
                else:graphs[name]=ab_graph(lambda ns=ns,st=st:ns['kernel_fn'](data,st))
            for seed in BR_SEEDS:
                assert time.monotonic()<deadline-10,'budget during seeds'
                if seed!=BR_SEEDS[0]:ab_copy(data,mk(seed))
                for graph,_ in graphs.values():graph.replay()
                torch.cuda.synchronize()
                for name,(_,out) in graphs.items():
                    if name=='A':continue
                    qual=pa_quality(out,graphs['A'][1]);old=quality.setdefault(name,dict(id_rows=0,l2=0.));old['id_rows']=max(old['id_rows'],qual['id_rows']);old['l2']=max(old['l2'],qual['l2'])
                    if why:=ab_gate(out,graphs['A'][1]):gates.setdefault(name,why)
                values=ab_time(graphs,BR_ROUNDS);samples.append({n:statistics.mean(v) for n,v in values.items()})
            unprofiled={n:statistics.median(v) for n,v in ab_time({n:graphs[n] for n in saved},5).items()}
            for name,(mapping,_) in saved.items():profiles[name]=dc_profile(graphs[name][0],mapping,act)
            io=saved['A'][1];io['key']=key;ready=br_ready(arms,io)
            # Every graph/call has private outputs. Producer flags live in its captured pool.
            cases.append(dict(case=('d' if q==1 else 'v')+str(b)+('S' if window else 'F'),samples=samples,gates=gates,quality=quality,profiles=profiles,dag=dag,ready=ready,unprofiled_event_us=unprofiled,plan=state['best7_plans'][key]))
            del data,graphs,saved,io;gc.collect();torch.cuda.empty_cache()
    result=dict(base_sha256=hashlib.sha256(PA_BASE.encode()).hexdigest(),source_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in BR_SOURCES.items()},seeds=list(BR_SEEDS),rounds=BR_ROUNDS,reused_jit=reused,cases=cases)
    blob=br_pack(result);lines=['META S14;24seeds*8coldREAD;A2=A:A;ratios1e-4,A0.01us',
        'META M/E/S/R/C/P/E1/E2;DAG+roles+stagebounds;ready A/M/E/R',
        'BR_REBUILD_XZ85 '+blob,'END elapsed='+str(round(time.monotonic()-started,1))]
    assert sum(len(s)+1 for s in lines)<9000,'archive exceeds receipt cap'
    return lines
