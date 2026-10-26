MT_SEEDS=tuple(8101613+104729*i for i in range(32))
MT_REPS=8

def mt_pair(a,b):
    values={'A':[],'B':[]};start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for rep in range(MT_REPS):
        for name,graph in ((('A',a),('B',b)) if rep%2==0 else (('B',b),('A',a))):
            lab_cold_l2();start.record();graph.replay();end.record();end.synchronize();values[name].append(start.elapsed_time(end)*1000)
    ratios=[b/a for a,b in zip(values['A'],values['B'])]
    return dict(ratio=statistics.mean(ratios),a_us=statistics.mean(values['A']),b_us=statistics.mean(values['B']),round_sd=statistics.stdev(ratios))

def mt_pack(result):
    binary=[]
    for cell in result['cases']:
        samples=cell.pop('samples');arms=list(samples[0]);cell['columns']=arms;cell['sample_rows']=len(samples);cell['sample_offset']=len(binary)
        cell['measured_means']={a:{k:round(statistics.mean(row[a][k] for row in samples),3) for k in ('a_us','b_us')} for a in arms}
        cell['aa_round_sd_mean']=round(statistics.mean(row['A2']['round_sd'] for row in samples),6)
        for row in samples:
            for arm in arms:
                v=round((row[arm]['ratio']-1)*100000);binary.append((v<<1)^(v>>63))
                d=round((row[arm]['a_us']-cell['measured_means'][arm]['a_us'])*100);binary.append((d<<1)^(d>>63))
    result['version']=1;result['binary_count']=len(binary);result['ratio_resolution']=1e-5
    text=json.dumps(result,separators=(',',':')).encode();raw=bytearray(struct.pack('<I',len(text))+text)
    for v in binary:
        while v>=128:raw.append((v&127)|128);v>>=7
        raw.append(v)
    return base64.b85encode(lzma.compress(raw,preset=9)).decode()

def mt_profile_summary(p,dag):
    trace=p['trace'];by={n['node']:n for n in trace};out={};heavy=[]
    for n in trace:
        name=n['name'];role=n['role']
        if role=='moe':
            if 'route' in name:label='route_meta'
            elif 'finish' in name:label='finish'
            elif 'gemm<32' in name:label='heavy_w13' if 'true' in name.split('>(')[0] else 'heavy_w2'
            elif 'gemm<16' in name:label='light_w13' if 'true' in name.split('>(')[0] else 'light_w2'
            else:label='small_w13' if 'true' in name.split('>(')[0] else 'small_w2'
            out[label]=dict(start_us=round(n['start_ns']/1000,3),end_us=round(n['end_ns']/1000,3),grid=n['grid'],block=n['block'],shared=n['shared'],registers=n['registers'],implementation=name.split('(')[0][:48])
            if label.startswith(('heavy_','light_')):heavy.append([name,n['grid'],n['block'],n['shared'],n['registers']])
    finishes={i for i,n in enumerate(dag['nodes']) if 'finish' in by.get(n[0],{}).get('name','')}
    out['finish_edges']=[[by.get(dag['nodes'][a][0],{}).get('name','join')[:24],typ] for a,b,typ,*_ in dag['edges'] if b in finishes]
    out={k:v for k,v in out.items() if k in ('route_meta','finish','finish_edges') or k.endswith('_w2')}
    for k in ('heavy_w2','light_w2','small_w2'):
        if k in out:out[k]=dict(end_us=out[k]['end_us'])
    return out,sorted(heavy,key=lambda row:row[0])

def lab(deadline,weights,state=None):
    assert weights is not None and state is not None,'exact S15 decode host prepare phase required'
    begun=time.monotonic();ns=t15_namespace();ns['nf_module']=kernel_fn.__globals__['nf_module'];exec(compile(MT_RUNTIME,'<p7-tail-runtime>','exec'),ns)
    ns['mt_module']();ns['mt_pmz_module']();act=t15_activity();cases=[]
    for b in (16,32):
      for window in (2048,0):
        assert time.monotonic()<deadline-15,'budget before shape'
        mk=lambda seed:ns['_ap_generate_input'](batch=b,ctx=80000,layer='sliding' if window else 'full',q_len=8,spread='agentic',dist='soft',page=1,seed=seed,device='cuda')
        ns['MT_ARM']='A';data=mk(MT_SEEDS[0]);t15_busiest(ns,data,state);graphs={};maps={};dags={};signatures={};gates={};samples=[];profiles={}
        for arm in ns['MT_ARMS']:
            ns['MT_ARM']=arm;ns['kernel_fn'](data,state)
        torch.cuda.synchronize()
        for arm in ns['MT_ARMS']:
            ns['MT_ARM']=arm;g,out,mapping,_,dag=t15_capture(ns['kernel_fn'],state,data,act,ns);graphs[arm]=(g,out);maps[arm]=mapping;dags[arm]=dag
        for i,seed in enumerate(MT_SEEDS):
            assert time.monotonic()<deadline-8,'budget during seeds'
            if i:
                ns['MT_ARM']='A';fresh=mk(seed);t15_busiest(ns,fresh,state);ab_copy(data,fresh);del fresh
            for g,_ in graphs.values():g.replay()
            torch.cuda.synchronize()
            for arm,(g,out) in graphs.items():
                if arm!='A' and (why:=ab_gate(out,graphs['A'][1])):gates.setdefault(arm,why)
            row={}
            others=list(ns['MT_ARMS'][1:]);others=others[i%len(others):]+others[:i%len(others)]
            for arm in others:row[arm]=mt_pair(graphs['A'][0],graphs[arm][0])
            samples.append(row)
        for arm in ns['MT_ARMS']:
            p=t15_profile(graphs[arm][0],maps[arm],dags[arm],act,window);profiles[arm],heavy=mt_profile_summary(p,dags[arm]);signatures[arm]=t15_signature(p['trace'],dags[arm])
            if arm=='A':baseline_heavy=heavy
            else:assert heavy==baseline_heavy,('heavy/light kernels changed',arm)
        assert signatures['A']==signatures['A2'],'A:A schedule/resource/DAG mismatch'
        assert all('route_meta' in profiles[a] and 'finish' in profiles[a] for a in profiles),'selected shell branch not reached'
        assert all('mt::route' in profiles[a]['route_meta']['implementation'] for a in ('R','C'))
        assert all('mt::warp_finish' in profiles[a]['finish']['implementation'] for a in ('F4','F8','C','PF'))
        assert 'mt::finish' in profiles['P']['finish']['implementation']
        # Symbols verify actual changed kernels rather than a dispatch that silently retained S15.
        for arm in ('P','PF'):
            assert any(t==1 for _,t in profiles[arm]['finish_edges']),('dependent finish edge not captured',arm,profiles[arm]['finish_edges'])
        profiles={a:p for a,p in profiles.items() if a in ('A','R','F4','E','P','PF')}
        cases.append(dict(case='v'+str(b)+('S' if window else 'F'),gates=gates,aa_schedule_identical=True,samples=samples,profiles=profiles))
        del graphs,maps,dags,data;gc.collect();torch.cuda.empty_cache()
    result=dict(runtime_sha256=hashlib.sha256(MT_RUNTIME.encode()).hexdigest(),base_sha256='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb',seeds=list(MT_SEEDS),reps=MT_REPS,arms=list(ns['MT_ARMS']),cases=cases,elapsed_s=round(time.monotonic()-begun,2))
    # Preserve each paired seed; rounding is below the measured event resolution.
    for cell in cases:
      for row in cell['samples']:
       for values in row.values():
        for k,v in values.items():values[k]=round(v,6 if k in ('ratio','round_sd') else 3)
    blob=mt_pack(result);lines=['META P7 S15 MoE shell;32 seeds;balanced AB/BA x8;identical A:A captures',
      'META R=warp route,F4/F8=warp finish,C=R+F4,E=W2 entry,P=E+PDL original,PF=E+PDL warp;heavy/light unchanged',
      'MOE_TAIL_XZ85 '+blob,'END elapsed='+str(result['elapsed_s'])]
    assert sum(len(s)+1 for s in lines)<9000,('shell receipt too large',len(blob))
    return lines
