
def nearby_namespace(name,source):
    path='<nearby-'+name+'>'
    linecache.cache[path]=(len(source),None,source.splitlines(True),path)
    ns={'__name__':'nearby_'+name}
    exec(compile(source,path,'exec'),ns)
    return ns


def nearby_copy(dst,src):
    for k,v in src.items():
        if isinstance(v,torch.Tensor):
            if dst[k].shape!=v.shape:raise ValueError('fresh-input shape changed '+k)
            dst[k].copy_(v)


def nearby_graph(fn):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(stream)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn()
    g.replay();torch.cuda.synchronize()
    return g,out


def nearby_schema(out,data):
    t=data['config']['tokens']
    expected=(((t,8192),torch.bfloat16),((t,8192),torch.float32),((t,8),torch.int32))
    if not isinstance(out,(list,tuple)) or len(out)!=3:return 'wrong return structure'
    for v,(shape,dtype) in zip(out,expected):
        if tuple(v.shape)!=shape or v.dtype!=dtype:return 'wrong shape/dtype'
        if not bool(torch.isfinite(v).all()):return 'nonfinite'
    return None


def nearby_delta(a,b):
    # Diagnostic only: baseline's legal FP8-query rounding is not an FP32 TASK oracle.
    return [round(float((x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-12)),5)
            for x,y in zip(a[:2],b[:2])]


def lab(deadline,weights,state):
    began=time.monotonic();base=kernel_fn.__globals__
    inputs=nearby_namespace('inputs',INPUT_SOURCE)
    if torch.cuda.get_device_capability()!=(9,0):inputs['_fa3_scheduler_metadata']=lambda *a,**kw:None
    baseline=nearby_namespace('baseline',BASELINE_SOURCE)
    # The real baseline cannot execute its FA3 binary on local sm120. Its exact
    # Python ABI is screened with the same explicit launch stand-in as S15.
    local=globals().get('NEARBY_LOCAL_FLASH')
    if local:baseline['flash_attn_with_kvcache']=local
    baseline_state=None;baseline_error=None
    try:baseline_state=baseline['prepare'](weights)
    except Exception as exc:baseline_error=type(exc).__name__+':'+str(exc)[:160]
    calls={'s15':lambda d:base['kernel_fn'](d,state)}
    if baseline_state is not None:calls['sglang']=lambda d:baseline['kernel_fn'](d,baseline_state)
    ev0=torch.cuda.Event(enable_timing=True);ev1=torch.cuda.Event(enable_timing=True)
    def replay(g):
        lab_cold_l2();ev0.record();g.replay();ev1.record();ev1.synchronize()
        return ev0.elapsed_time(ev1)*1000
    x=torch.zeros(1,device='cuda');empty,_=nearby_graph(lambda:x.add_(1))
    fixed=statistics.median(replay(empty) for _ in range(20))
    rows=[];errors=[]
    print('NEARBY_PROGRESS begin cases='+str(len(CASES)),flush=True)
    for spec in CASES:
        label=spec['label'];case={k:v for k,v in spec.items() if k!='label'}
        if time.monotonic()>deadline-5:
            errors.append(label+':deadline');continue
        graphs={};data=None;per=[];gates=[];diff=[];ranges=[]
        def make(seed):
            data=inputs['generate_input'](**case,seed=seed)
            # Select the busiest range using this input's actual stage routing;
            # no weights or routing outputs are cached across timed calls.
            out=calls['s15'](data)
            start=32*torch.bincount((out[2].long()//32).flatten(),minlength=8).argmax()
            data['local_start'].copy_(start.reshape(1).int())
            return data
        try:
            data=make(SEEDS[0])
            graphs={name:nearby_graph(lambda fn=fn:fn(data)) for name,fn in calls.items()}
            for seed in SEEDS:
                if time.monotonic()>deadline-3:raise TimeoutError('seed budget')
                if seed!=SEEDS[0]:nearby_copy(data,make(seed))
                ranges.append([int(data['seq_lens'].min()),int(data['seq_lens'].max())])
                for g,_ in graphs.values():g.replay()
                torch.cuda.synchronize()
                for name,(_,out) in graphs.items():
                    if reason:=nearby_schema(out,data):gates.append(name+':'+reason)
                if 'sglang' in graphs:diff.append(nearby_delta(graphs['s15'][1],graphs['sglang'][1]))
                times={name:[] for name in graphs};ratios=[]
                names=list(graphs)
                # Swapped order, paired on identical input/cache rows. The only
                # modified cache entries belong to this block and are overwritten
                # before use by either side on every call.
                for r in range(8):
                    for name in names if r%2==0 else names[::-1]:
                        times[name].append(max(.001,replay(graphs[name][0])-fixed))
                    if r%2 and 'sglang' in times:
                        ratios.append((times['sglang'][-2]+times['sglang'][-1])/
                                      (times['s15'][-2]+times['s15'][-1]))
                per.append(dict(us={k:statistics.median(v) for k,v in times.items()},
                                speedup=statistics.median(ratios) if ratios else None))
            row=dict(k=label,n=len(per),us=[round(statistics.median(p['us']['s15'] for p in per),2)],
                     seq=ranges,ok=not gates)
            if 'sglang' in graphs:
                rr=[p['speedup'] for p in per]
                row['us'].append(round(statistics.median(p['us']['sglang'] for p in per),2))
                row.update(r=round(statistics.median(rr),4),rr=[round(v,4) for v in rr],delta=diff)
            if gates:row['gates']=gates
            rows.append(row)
        except Exception as exc:
            errors.append(label+':'+type(exc).__name__+':'+str(exc)[:110])
        graphs=data=None;gc.collect();torch.cuda.empty_cache()
    result=dict(candidate=S15_SHA,shape_count=len(CASES),seeds=SEEDS,replays=8,
                fixed_us=round(fixed,3),baseline_error=baseline_error,rows=rows,errors=errors,
                elapsed=round(time.monotonic()-began,2),complete=len(rows)==len(CASES) and not errors)
    # Compact all rows into the wrapper's 9k tail budget, while retaining timing
    # and per-seed paired ratios. Detailed checks/ranges are in local evidence.
    compact=[]
    for row in rows:
        compact.append([row['k'],*row['us'],row.get('r'),row.get('rr'),row['ok']])
    result['rows']=compact
    text=json.dumps(result,separators=(',',':'))
    if len(text)>8400:raise ValueError('receipt exceeds 8400 chars '+str(len(text)))
    return ['META S15 nearby audit; us=[S15,SGLang]; r=SGLang/S15; two seeds,8 cold ABBA replays; no TASK oracle on remote',
            'RESULT '+text]
