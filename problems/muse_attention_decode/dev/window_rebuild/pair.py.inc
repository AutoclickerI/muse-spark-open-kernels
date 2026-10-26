SEEDS=tuple(22100027+104729*i for i in range(24))
ROUNDS=10
BATCHES=(16,32)
RESULTS=[]

def window_host():
    fn='<p5-window-paired-S15>';linecache.cache[fn]=(len(BASE),None,BASE.splitlines(True),fn)
    m=types.ModuleType('p5_window_pair_host');exec(compile(BASE,fn,'exec'),m.__dict__);return m

def fa3_eight(host,qkv,data,state):
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    q,g=host.qk_prep(qkv,data,state,True);c=data['config'];descale={}
    if c['kv_scale']!=1.0:descale=dict(k_descale=data['k_scale'].reshape(1,1).expand(c['batch'],host.HKV),v_descale=data['v_scale'].reshape(1,1).expand(c['batch'],host.HKV))
    a=flash_attn_with_kvcache(q=q,k_cache=data['k_cache'],v_cache=data['v_cache'],page_table=data['page_table'],cache_seqlens=data['cache_seqlens'],cu_seqlens_q=data['cu_seqlens_q'],max_seqlen_q=8,softmax_scale=c['sm_scale']*c['qk_scale_factor'],causal=True,window_size=(c['window']-1,0),scheduler_metadata=data['scheduler_metadata'],num_splits=8,**descale)
    if isinstance(a,tuple):a=a[0]
    o=torch.empty((c['tokens'],host.Q_SIZE),device=q.device,dtype=torch.bfloat16)
    host._launch(host._norm_gate_kernel,(c['tokens'],),a,g,o,host.RMS_EPS,HQ=host.HQ,D=host.HD,num_warps=4);return o

def window_arm(host,name):
    m=types.ModuleType('p5_window_'+name);g=host.__dict__;v=m.__dict__;v.update(g)
    for k,f in g.items():
        if isinstance(f,types.FunctionType) and f.__globals__ is g:
            q=types.FunctionType(f.__code__,v,f.__name__,f.__defaults__,f.__closure__);q.__kwdefaults__=f.__kwdefaults__;q.__dict__.update(f.__dict__);v[k]=q
    def call(qkv,data,state,plan):
        c=data['config']
        if c['q_len']!=8 or not c['window']:return host._best7_attention(qkv,data,state,plan)
        if name=='F8':
            return fa3_eight(host,qkv,data,state) if c['batch']==32 else host._best7_attention(qkv,data,state,plan)
        arm='STREAM2' if name=='RAW2' else name
        cfg=dict(c,num_q_heads=host.HQ,num_kv_heads=host.HKV,head_dim=64)
        return wr_attention(dict(data,config=cfg),qkv,arm,name!='RAW2').view(-1,host.Q_SIZE)
    v['_best7_attention']=call;return m

def metadata_probe(host,data):
    if globals().get('LOCAL_STANDIN'):return dict(local_only=True)
    from sgl_kernel.flash_attn import get_scheduler_metadata
    c=data['config'];meta={}
    for splits in (0,1,2,4,8):
        x=get_scheduler_metadata(batch_size=c['batch'],max_seqlen_q=8,max_seqlen_k=data['page_table'].shape[1]*c['page_size'],num_heads=host.HQ,num_heads_k=host.HKV,headdim=64,
            cache_seqlens=data['cache_seqlens'],qkv_dtype=torch.float8_e4m3fn,cu_seqlens_q=data['cu_seqlens_q'],page_size=1,causal=True,window_size=(2047,0),num_splits=splits)
        meta[str(splits)]=dict(shape=list(x.shape),values=x.cpu().tolist())
    return dict(values=meta,note='Untimed introspection only; no cached derived metadata is passed to a timed arm. First B int32 entries are dynamic split counts in the public FA3 layout; retain all raw entries for native-version audit.')

def clock_probe(host,st,data,name):
    c=data['config'];B=c['batch'];cfg=dict(c,num_q_heads=host.HQ,num_kv_heads=host.HKV,head_dim=64)
    clocks=torch.zeros((B*host.HKV*16,10),device='cuda',dtype=torch.int64)
    plan=st['best7_plans'][(B,8,2048)];role='qkv3' if host._gate_split_on(data,st,plan) else 'qkv_gate'
    def call():
        qkv=host._best7_project(data['attn_input'],role,st,plan)
        return wr_attention(dict(data,config=cfg),qkv,name+'_CLOCK',True,clocks)
    call();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=call()
    times=[]
    for _ in range(3):clocks.zero_();lab_cold_l2();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();times.append(a.elapsed_time(b)*1000)
    rows=clocks.cpu().tolist();rows=[r for r in rows if r[0]];origin=min(r[0] for r in rows)
    def summary(v):
        v=sorted(v)
        return dict(n=len(v),min=v[0],median=statistics.median(v),p90=v[min(len(v)-1,len(v)*9//10)],max=v[-1]) if v else None
    phases={str(i):summary([r[i]-origin for r in rows if r[i]]) for i in range(9)}
    order=(0,1,4,5,2,3,6,7,8) if name.startswith('STREAM') else tuple(range(9))
    deltas={str(a)+'->'+str(b):summary([r[b]-r[a] for r in rows if r[a] and r[b]]) for a,b in zip(order,order[1:])}
    return dict(name=name,phase_ns=phases,deltas_ns=deltas,graph_event_us=times,
        phase_labels=['metadata_entry','history_issued','copy_ready_or_first_tile_ready','transpose_done_or_first_tile_transpose','PDL_wait_done','Q_ready','history_and_new_tile_math_done','partials_published','merge_gate_done'],
        note='Instrumented source has different registers; phases diagnose ordering/critical work, not uninstrumented speed. Streaming only its initial tiles are prefetched before wait.',sample_rows=[[x-origin if x else 0 for x in r[:9]] for r in rows[:4]])

def lab(deadline,weights,state=None):
    global RESULTS
    started=time.monotonic();RESULTS=[]
    if weights is None:weights=PUBLIC_SPEC['generate_weights'](5353,device='cuda',router_seed=197731)
    host=window_host();st=host.prepare(weights);act=tl_activity()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(wr_module,wr_KERNELS))
    for B in BATCHES:
        names=('A','A2','F8','WG2C2','WG2C4','STREAM2','STREAM4','RAW2');variants={n:host if n in ('A','A2') else window_arm(host,n) for n in names}
        samples={n:[] for n in names};passed={n:0 for n in names};profiles={};resources={n:wr_module(n).sliding_resources(8,1,128,wr_TOPOLOGY[n]) for n in wr_TOPOLOGY}
        assert all(v[-1]>0 for v in resources.values()),('native cluster admission',B,resources)
        mk=lambda seed:host._ap_generate_input(batch=B,ctx=80000,layer='sliding',q_len=8,spread='agentic',dist='soft',page=1,seed=seed,device='cuda')
        for si,seed in enumerate(SEEDS):
            if time.monotonic()>deadline-12:raise TimeoutError(('incomplete window pair',B,si))
            d=mk(seed);graphs={};out={};offset=si%len(names)
            for n in names[offset:]+names[:offset]:
                for _ in range(2):variants[n].kernel_fn(d,st)
                torch.cuda.synchronize()
                if si==0 and n!='A2':graphs[n],out[n],p=cs_profile(variants[n],st,d,act);profiles[n]=p
                else:graphs[n],out[n],_,_=tl_capture(variants[n],st,d,act,False)
            checker=eh_checker(d,weights)
            for n in names:
                graphs[n].replay();torch.cuda.synchronize();ok,msg=checker(out[n]);assert ok,(B,n,seed,msg);passed[n]+=1
            del checker
            per={n:[] for n in names}
            for j in range(ROUNDS):
                order=names[j%len(names):]+names[:j%len(names)]
                if j&1:order=order[::-1]
                for n in order:per[n].extend(tl_times(graphs[n],1))
            for n in names:samples[n].append(statistics.mean(per[n]))
            if si==0:
                extra=dict(metadata=metadata_probe(host,d),clocks=[clock_probe(host,st,d,n) for n in ('WG2C2','STREAM2')])
            graphs.clear();out.clear();del d;gc.collect();torch.cuda.empty_cache()
        RESULTS.append(dict(case='v'+str(B)+'S',seeds=list(SEEDS),rounds=ROUNDS,time_unit='ns',
            times={n:[round(x*1000) for x in v] for n,v in samples.items()},TASK=passed,resources=resources,
            critical={n:{k:p['inc'].get(k,0) for k in ('qkv3','qkv_gate','attn','gate','o')} for n,p in profiles.items()},
            bounds={n:{k:{a:round(v[a]*1000) for a in ('start','end')} for k,v in p['stages'].items() if k in ('qkv3','qkv_gate','attn','gate','o')} for n,p in profiles.items()},
            nodes={n:[dict(name=z['name'][:200],start=round(z['start']*1000),end=round(z['end']*1000)) for z in p['nodes']] for n,p in profiles.items()},extra=extra))
    archive=dict(s15_sha256=hashlib.sha256(BASE.encode()).hexdigest(),results=RESULTS,elapsed_s=time.monotonic()-started)
    blob=base64.b85encode(lzma.compress(json.dumps(archive,separators=(',',':')).encode())).decode()
    if len(blob)>8500:
        for r in archive['results']:
            for c in r['extra']['clocks']:c.pop('sample_rows')
        blob=base64.b85encode(lzma.compress(json.dumps(archive,separators=(',',':')).encode())).decode()
    assert len(blob)<8600,('pair archive would truncate',len(blob))
    return ['META P5 structural window pair vs exact S15: A/A2, FA3split8(v32only),2WG residentC2/C4,streaming deferred-new-KVC2/C4,RAW2 same stream without early edge.',
            'META 24 disjoint seeds10cold READ rounds; fresh pools/rotated capture; every arm/seed public whole-layer TASK; per-phase clocks and actual FA3 split metadata are untimed diagnostics.',
            'WINDOW_PAIR_XZ85 '+blob,'END shapes='+str(len(RESULTS))+' elapsed='+str(round(time.monotonic()-started,2))]
