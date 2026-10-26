"""S20 vs S19 (INC), leave-outs and v32S alternatives restricted to the shapes they change.

The S19 module executes inside the payload (any problem can host it); a restricted arm runs only at its
rows and takes A2's time elsewhere, where its code is identical to A's."""
import ast,gc,hashlib,json,linecache,lzma,base64,types


def s16_copy_state(value):
    if isinstance(value,dict):return {k:s16_copy_state(v) for k,v in value.items()}
    if isinstance(value,list):return [s16_copy_state(v) for v in value]
    if isinstance(value,tuple):return tuple(s16_copy_state(v) for v in value)
    return value


def s16_source(arm):
    lines=S16_BUNDLE['host'].splitlines(True)
    for a,b,text in reversed(arm['edits']):lines[a:b]=[text]
    source=''.join(lines);assert hashlib.sha256(source.encode()).hexdigest()==arm['sha256']
    return source


def s16_same_jit(base,target):
    names={name for name,v in base.items() if hasattr(v,'warmup') and isinstance(getattr(v,'src',None),str)
           and getattr(target.get(name),'src',None)==v.src}
    deps={name:{n.id for n in ast.walk(ast.parse(base[name].src)) if isinstance(n,ast.Name) and hasattr(base.get(n.id),'warmup')} for name in names}
    while True:
        kept={name for name in names if deps[name]<=names}
        if kept==names:break
        names=kept
    for name in names:target[name]=base[name]
    return len(names)


def s16_namespace(name,source,base):
    path='<s16-arm-'+name+'-'+hashlib.sha256(source.encode()).hexdigest()[:12]+'>'
    linecache.cache[path]=(len(source),None,source.splitlines(True),path)
    m=types.ModuleType('s16_'+name);m.__file__=path;exec(compile(source,path,'exec'),m.__dict__)
    if globals().get('S16_LOCAL_INSTALL'):S16_LOCAL_INSTALL(m)
    v=m.__dict__
    def assignments(s):
        return {t.id:ast.dump(n) for n in ast.parse(s).body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}
    host,target=assignments(S16_BUNDLE['host']),assignments(source)
    # Share native caches only if every native source they can depend on matches.
    # This avoids silently using host PDL/epilogue code in its baseline off arm.
    native_identical=all(target.get(key)==value for key,value in host.items()
                          if 'CUDA' in key or 'CPP' in key)
    for key,code in target.items():
        if host.get(key)!=code or key not in base or callable(base[key]):continue
        if any(word in key.upper() for word in ('MOD','EXT','OPTS')) and not native_identical:continue
        if key in ('_GEMV_CHOICES','_s16_native_builds','NF_PORT_SELECTED','_VQ8_ROWS'):continue
        v[key]=base[key]
    assert all(v[key]==base[key] for key in ('GB_ROWS','GATE_SPLIT_ROWS','GATE_SPLIT_QKV3'))
    v['_GEMV_CHOICES']=dict(base['_GEMV_CHOICES'])
    return v,s16_same_jit(base,v)


S19_HOST=None
def s19_host():
    global S19_HOST
    source=S16_BUNDLE['host']+S19_HOST_HOOK
    path='<s19-host-'+hashlib.sha256(source.encode()).hexdigest()[:12]+'>'
    linecache.cache[path]=(len(source),None,source.splitlines(True),path)
    m=types.ModuleType('s19_host');m.__file__=path;exec(compile(source,path,'exec'),m.__dict__)
    if globals().get('S16_LOCAL_INSTALL'):S16_LOCAL_INSTALL(m)
    S19_HOST=m.__dict__;return S19_HOST


def lab(deadline,weights,state):
    started=time.monotonic();base=s19_host()
    # Independent unchanged public TASK criteria, never used by a candidate or
    # timed graph. Host-equivalence is diagnostic for arithmetic replacements.
    checker={};exec(compile(S16_CHECKER,'<s16-independent-TASK>','exec'),checker)
    if weights is None:
        # Import-phase experiment has its own held-out weights and state. The
        # judge subsequently prepares its real weights from scratch. No lab
        # output/state feeds the official timed candidate.
        weights=checker['generate_weights'](S17_WEIGHT_SEED,device='cuda',router_seed=S17_ROUTER_SEED)
        import io,contextlib
        log=io.StringIO();t0=time.monotonic()
        try:
            with contextlib.redirect_stdout(log):state=base['_candidate_prepare'](weights)
        except Exception as exc:
            # The run's answer is then S19's own prepare timeline: where the time went and what ran out.
            info=dict(error=type(exc).__name__+': '+str(exc)[:200],host_prepare_s=round(time.monotonic()-t0,1),build_s=dict(base.get('_S19_BUILD_S',{})),phase_s=dict(base.get('_S19_PHASE_S',{})),
                      prepare_lines=[l[:600] for l in log.getvalue().splitlines() if l.startswith(('BEST_PREPARE','BEST_PHASES','BEST6_OPTIONAL','VDSK_WARM_REJECT'))][-12:])
            return ['META S20 host prepare failed before the pair','S20_PREPARE '+json.dumps(info,separators=(',',':')),'END complete=False']
        host_prepare_s=time.monotonic()-t0
        prepare_lines=[l[:600] for l in log.getvalue().splitlines() if l.startswith(('BEST_PREPARE','BEST_PHASES','BEST6_OPTIONAL'))][-6:]
    device=weights['qkv_gate'].device
    original=base['_S16_LAB_BASE_STATE'];sources={'A':S16_BUNDLE['host']}
    ns={'A':base};states={'A':state};shared={};failures=[];agreements={};task_checks=0;arm_task={};arm_nearby={}
    for name,spec in S16_BUNDLE['arms'].items():
        source=s16_source(spec);sources[name]=source
        try:
            v,shared[name]=s16_namespace(name,source,base);ns[name]=v
            v['_vq8_module']();v['nf_module']();v['cd_module']()
            v['smallm_module']();v['_glue_cl_ext']();v['partial_port_ext'](False)
            st=s16_copy_state(original)
            st['vq8_rows']=dict(v['_VQ8_ROWS'])
            st['partial_port_rows']=v['_PP_ROWS']
            st['cluster_rows']=v.get('_CLUSTER_ROWS',())
            # The host's base prepare built r2_rows from S18's _R2_SELECTION (D8/D4);
            # S17's delta never rebuilds them, so INC would otherwise time S18's rows.
            if '_r2_prepare' in v:st['r2_rows']=v['_r2_prepare'](device)
            if '_s16_prepare_delta' in v:st=v['_s16_prepare_delta'](weights,st)
            states[name]=st
        except Exception as exc:failures.append(dict(scope='arm',arms=[name],reason='prepare:'+type(exc).__name__+':'+str(exc)[:100]))
    generator=base['_ap_generate_input']
    if torch.cuda.get_device_capability(device)!=(9,0):generator=ab_namespace({'_ap_fa3_scheduler_metadata':lambda *a,**kw:None})['_ap_generate_input']
    # Native current-composition TASK checks at every review case. These are
    # separate from score timing and never feed candidate state or output.
    nearby_task_checks=0;nearby_expected_checks=len(S17_NEARBY_CASES)*2*len(S17_NEARBY_ARMS)
    if torch.cuda.get_device_capability(device)!=(9,0):checker['_fa3_scheduler_metadata']=lambda *a,**kw:None
    for spec in S17_NEARBY_CASES:
        label=spec['label'];case={k:v for k,v in spec.items() if k!='label'}
        if time.monotonic()>deadline-8:failures.append('nearby:'+label+':deadline');break
        data=graphs=None
        try:
            make=lambda seed:checker['generate_input'](**case,seed=seed,device=device)
            data=make(AB_SEEDS[0])
            graphs={name:ab_graph(lambda v=ns[name],st=states[name]:v['kernel_fn'](data,st)) for name in S17_NEARBY_ARMS}
            for seed in AB_SEEDS[:2]:
                if seed!=AB_SEEDS[0]:ab_copy(data,make(seed))
                ff,bf=checker['ref_front'],checker['ref_back'];fc={};bc={}
                def front(d,w,query_dtype=torch.bfloat16):
                    if query_dtype not in fc:fc[query_dtype]=ff(d,w,query_dtype)
                    return fc[query_dtype]
                def back(f,w,ids):
                    key=id(f),tuple(ids.reshape(-1).tolist())
                    if key not in bc:bc[key]=bf(f,w,ids)
                    return bc[key]
                checker['ref_front'],checker['ref_back']=front,back
                try:
                    for name,(graph,output) in graphs.items():
                        graph.replay();graph.replay();torch.cuda.synchronize()
                        ok,message=checker['check_implementation'](data,checker['collect'](data,output),weights);nearby_task_checks+=1
                        arm_nearby[name]=arm_nearby.get(name,0)+1
                        if not ok:failures.append(dict(scope='arm',arms=[name],reason='nearby:'+label+':TASK:'+message))
                finally:checker['ref_front'],checker['ref_back']=ff,bf
                fc=bc=None
        except Exception as exc:failures.append('nearby:'+label+':'+type(exc).__name__+':'+str(exc)[:160])
        graphs=data=None;gc.collect();torch.cuda.empty_cache()

    times={};shapes={}
    for b,q,w in AB_CASES:
        key=(b,q,w);label=ab_label(key)
        if time.monotonic()>deadline-8:failures.append(label+':deadline');continue
        make=lambda seed:generator(batch=b,q_len=q,ctx=80000,layer='sliding' if w else 'full',spread='fixed' if b==1 else 'agentic',dist='soft',page=1,seed=seed,device=device)
        data=make(AB_SEEDS[0]);graphs={};gates={}
        for name,v in dict(A=base,A2=base,**{n:ns[n] for n in states if n!='A'}).items():
            if S16_BUNDLE['arms'].get(name,{}).get('rows') and label not in S16_BUNDLE['arms'][name]['rows']:continue
            st=state if name=='A2' else states[name]
            try:graphs[name]=ab_graph(lambda v=v,st=st:v['kernel_fn'](data,st))
            except Exception as exc:gates[name]='capture:'+str(exc)[:100]
        if 'A' not in graphs:failures.append(label+':host capture');continue
        per=[]
        for seed in AB_SEEDS:
            if time.monotonic()>deadline-4:failures.append(label+':seeds incomplete');break
            if seed!=AB_SEEDS[0]:ab_copy(data,make(seed))
            for graph,_ in graphs.values():graph.replay()
            torch.cuda.synchronize()
            front_fn=checker['ref_front'];back_fn=checker['ref_back'];front_cache={};back_cache={}
            def front(d,w,query_dtype=torch.bfloat16):
                assert d is data and w is weights
                if query_dtype not in front_cache:front_cache[query_dtype]=front_fn(d,w,query_dtype)
                return front_cache[query_dtype]
            def back(f,w,ids):
                key=id(f),tuple(ids.reshape(-1).tolist())
                if key not in back_cache:back_cache[key]=back_fn(f,w,ids)
                return back_cache[key]
            checker['ref_front']=front;checker['ref_back']=back
            try:
                for name,(_,output) in graphs.items():
                    if name!='A' and (why:=ab_gate(output,graphs['A'][1])):agreements.setdefault(label+':'+name,why)
                    ok,message=checker['check_implementation'](data,checker['collect'](data,output),weights);task_checks+=1;arm_task[name]=arm_task.get(name,0)+1
                    if not ok:gates.setdefault(name,'TASK:'+message)
            finally:checker['ref_front']=front_fn;checker['ref_back']=back_fn
            front_cache=back_cache=None
            values=ab_time(graphs,16 if b*q<=8 else 8)
            row={name:statistics.mean(v) for name,v in values.items()}
            for name in states:row.setdefault(name,row['A2'])  # restricted arm: code identical to A here
            per.append(row)
        for name,reason in gates.items():failures.append(dict(scope='arm',arms=[name],reason=label+':'+reason))
        if per:
            times[key]=per;aa=[s['A2']/s['A'] for s in per]
            shapes[label]=dict(n=len(per),aa=[min(aa),max(aa)],r={name:statistics.mean(s[name]/s['A'] for s in per) for name in graphs if name!='A'})
        graphs=data=None;gc.collect();torch.cuda.empty_cache()
    loads={}
    for (b,q),label in AB_LOADS.items():
        if not all((b,q,w) in times for w in (2048,0)):continue
        pairs=list(zip(times[b,q,2048],times[b,q,0]));aa=[(46*s['A2']+16*f['A2'])/(46*s['A']+16*f['A']) for s,f in pairs]
        # Every paired seed survives. Means/SE/normalized summaries duplicate
        # these samples and can be recomputed by the receipt reader.
        loads[label]=dict(n=len(pairs),aa=[min(aa),max(aa)],samples=[{name:(46*s[name]+16*f[name])/(46*s['A']+16*f['A']) for name in s} for s,f in pairs])
    arms=S16_BUNDLE['arms']
    active={ab_label(k):2+sum(1 for n,s in arms.items() if n in states and (not s.get('rows') or ab_label(k) in s['rows'])) for k in AB_CASES}
    expected_task=16*sum(active.values())
    plan=state.get('attention_plan',{})
    result=dict(host_sha256=hashlib.sha256(S16_BUNDLE['host'].encode()).hexdigest(),shapes=shapes,loads=loads,
                failures=failures,shared_jit=shared,seeds=len(AB_SEEDS),seed_ids=list(AB_SEEDS),task_checks=task_checks,expected_task_checks=expected_task,
                nearby_task_checks=nearby_task_checks,nearby_expected_checks=nearby_expected_checks,agreement_diagnostics=agreements,
                host_prepare_s=host_prepare_s,prepare_lines=prepare_lines,phase_s=dict(base.get('_S19_PHASE_S',{})),build_s=dict(base.get('_S19_BUILD_S',{})),delta_s=dict(state.get('s19_prepare_s',{})),
                attention_plan={str(k):str(plan.get(k)) for k in ((16,1,False),(32,1,False),(16,8,True),(32,8,True))},
                checker_sha256=hashlib.sha256(S16_CHECKER.encode()).hexdigest(),elapsed=time.monotonic()-started,left=deadline-time.monotonic())
    result['complete']=(not failures and len(shapes)==12 and len(loads)==6 and task_checks==expected_task and len(states)==len(arms)+1
                        and nearby_task_checks==nearby_expected_checks and all(s['n']==16 and len(s['r'])==active[l]-1 for l,s in shapes.items()))
    failures=[f if isinstance(f,dict) else dict(scope='run',arms=[],reason=f) for f in failures]
    result['failures']=failures
    native=torch.cuda.get_device_capability(device)==(9,0)
    identities={'A':result['host_sha256'],'A2':result['host_sha256'],**{n:a['sha256'] for n,a in arms.items()}}
    result['arm_status']={}
    for name,identity in identities.items():
        expected=len(arms.get(name,{}).get('rows') or AB_CASES)*len(AB_SEEDS)
        faults=[f for f in failures if not f['arms'] or name in f['arms']]
        result['arm_status'][name]=dict(sha256=identity,native=native,
            complete=not faults and len(loads)==6 and all(v['n']==len(AB_SEEDS) for v in loads.values()),
            seed_ids=list(AB_SEEDS),task_checks=arm_task.get(name,0),expected_task_checks=expected,
            nearby_task_checks=arm_nearby.get(name,0),nearby_expected_checks=130 if name in S17_NEARBY_ARMS else 0,failures=faults)
    import struct
    names=list(next(iter(loads.values()))['samples'][0]) if loads else [];labels=list(loads)
    values=[round(row[name]*1000000)-1000000 for label in labels for row in loads[label]['samples'] for name in names]
    result['sample_arms']=names;result['sample_loads']=labels;result['sample_scale']=1000000
    result['sample_i32_b85']=base64.b85encode(struct.pack('<'+'i'*len(values),*values)).decode()
    for cell in result['loads'].values():cell.pop('samples')
    def compact(value):
        if isinstance(value,float):return round(value,6)
        if isinstance(value,dict):return {k:compact(v) for k,v in value.items()}
        if isinstance(value,list):return [compact(v) for v in value]
        return value
    blob=base64.b85encode(lzma.compress(json.dumps(compact(result),separators=(',',':')).encode())).decode()
    assert len(blob)<8500,len(blob)
    return ['META S20 on frozen S18;16 disjoint held-out seeds,fresh weights/router;public TASK all arms;restricted arms take A2 off their rows;S19 executed in-payload',
            'S20_XZ85 '+blob,'END complete='+str(result['complete'])]
