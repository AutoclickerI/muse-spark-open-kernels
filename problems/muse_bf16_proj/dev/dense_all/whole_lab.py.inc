def whole_load():
    path='<P10 whole>';linecache.cache[path]=(len(WHOLE_SOURCE),None,WHOLE_SOURCE.splitlines(True),path)
    module=types.ModuleType('da_whole');module.__file__=path;exec(compile(WHOLE_SOURCE,path,'exec'),module.__dict__)
    if globals().get('WHOLE_LOCAL_INSTALL'):WHOLE_LOCAL_INSTALL(module)
    return module.__dict__

def whole_early(m,bank):
    selection=m['DA_SELECTED'];points={(t,r):3 for (t,r),a in selection.items() if t<=32 and a==0 and r!='gate_apply'}
    req={(t,bank['roles']['unpooler_slice' if r=='unpooler' and t<=8 else r].shape[1],3) for t,r in points}
    if req:
        mod,index,source=m['da_build_small'](bank['ns'],True,False,m['DA_LOCAL_REDUCE'],req)
        for (t,r),a in points.items():
            role='unpooler_slice' if r=='unpooler' and t<=8 else r;p=bank['small'][t,role,0];idx,cfg=index[t,p['cfg'][1]*p['cfg'][2]*p['cfg'][3]*64,3]
            bank['small'][t,role,3]=dict(p,idx=idx,cfg=cfg,fn=mod.v18s_launch)
    # Only a schedule-equivalent early arm is compared; no additional encoding.
    return {**selection,**points}

def lab(deadline,weights,state=None):
    import contextlib,io,gc
    start=time.monotonic();task={}
    for i,src in enumerate(WHOLE_TASK):exec(compile(src,'<published layer TASK '+str(i)+'>','exec'),task)
    weights=task['generate_weights'](14000311 if WHOLE_CONFIRM else 12000311,router_seed=14000313 if WHOLE_CONFIRM else 12000313);m=whole_load();log=io.StringIO();t0=time.monotonic()
    with contextlib.redirect_stdout(log):st=m['prepare'](weights)
    torch.cuda.synchronize();prepare_s=time.monotonic()-t0
    selection=dict(WHOLE_SELECTION);early=whole_early(m,st['da']) if selection else {};added_early=dict(m.get('DA_TIMES',{}));seeds=[(14000319 if WHOLE_CONFIRM else 12000319)+104729*i for i in range(16 if WHOLE_CONFIRM else 8)]
    arms=(['A','AA','CODE'] if WHOLE_CONFIRM else ['A','AA','CODE','EARLY','MERGE','BOTH']) if selection else ['A','AA','MERGE'];rows=[];failure=None;checks=0
    if torch.cuda.get_device_capability()!=(9,0):task['_fa3_scheduler_metadata']=lambda *a,**kw:None
    def invoke(arm,data):
        m['DA_DO_MERGE']=arm in ('MERGE','BOTH');m['DA_DO_CODE']=arm in ('CODE','EARLY','BOTH')
        if selection:m['DA_SELECTED']=early if arm=='EARLY' else selection
        return m['kernel_fn'](data,st)
    try:
        for b,q,w in AB_CASES:
            if time.monotonic()>deadline-15:raise TimeoutError('before '+str((b,q,w)))
            if globals().get('WHOLE_LOCAL_CASES') and (b,q,w) not in WHOLE_LOCAL_CASES:continue
            make=lambda seed:task['generate_input'](batch=b,q_len=q,ctx=80000,layer='sliding' if w else 'full',spread='fixed' if b==1 else 'agentic',dist='soft',page=1,seed=seed,device='cuda')
            data=make(seeds[0]);graphs={arm:ab_graph(lambda arm=arm:invoke(arm,data)) for arm in arms};values={arm:[] for arm in arms}
            for seed in seeds:
                if time.monotonic()>deadline-8:raise TimeoutError('seed '+str((b,q,w,seed)))
                if seed!=seeds[0]:ab_copy(data,make(seed))
                for g,out in graphs.values():g.replay()
                torch.cuda.synchronize();front_fn,back_fn=task['ref_front'],task['ref_back'];fc={};bc={}
                def front(d,wt,query_dtype=torch.bfloat16):
                    if query_dtype not in fc:fc[query_dtype]=front_fn(d,wt,query_dtype)
                    return fc[query_dtype]
                def back(f,wt,ids):
                    key=id(f),tuple(ids.reshape(-1).tolist())
                    if key not in bc:bc[key]=back_fn(f,wt,ids)
                    return bc[key]
                task['ref_front'],task['ref_back']=front,back
                try:
                    for arm,(g,out) in graphs.items():
                        ok,msg=task['check_implementation'](data,task['collect'](data,out),weights);checks+=1
                        if not ok:raise AssertionError(str((b,q,w,seed,arm,msg)))
                finally:task['ref_front'],task['ref_back']=front_fn,back_fn
                ts=ab_time(graphs,12 if b*q<=8 else 8)
                for arm,samples in ts.items():values[arm].append(round(statistics.median(samples),4))
            rows.append(dict(case=[b,q,w],values=values));graphs=data=None;gc.collect();torch.cuda.empty_cache()
    except Exception as error:
        import traceback
        failure=repr(error)+' | '+' | '.join(traceback.format_exc().splitlines()[-5:])
    archive=dict(base_sha256=hashlib.sha256(WHOLE_SOURCE.encode()).hexdigest(),seeds=seeds,rows=rows,checks=checks,prepare_s=prepare_s,added_prepare=st.get('da_prepare_s',{}),early_prepare=added_early,elapsed_s=time.monotonic()-start,failure=failure,complete=len(rows)==12 and checks==12*len(seeds)*len(arms) and failure is None)
    if callable(globals().get('WHOLE_LOCAL_ARCHIVE')):WHOLE_LOCAL_ARCHIVE(archive)
    blob=base64.b85encode(lzma.compress(json.dumps(archive,separators=(',',':')).encode())).decode()
    assert len(blob)<8300,len(blob)
    return ['META P10 frozen S19 coding/early-load/merge;heldout seeds all12shapes;public TASK all arms;raw per-seed us retained','WHOLE_XZ85 '+blob,'END complete='+str(archive['complete'])]
