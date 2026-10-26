# Deliberately excluded-work oracle. Never used in a submission or promotion.
CASES=((1,1),(16,1),(32,1),(1,8),(16,8),(32,8))
SEEDS=tuple(85000121+104729*i for i in range(8))
ROUNDS=6
LOCAL=False

def cf_host():
    path='<P5-consumer-S19>';linecache.cache[path]=(len(BASE),None,BASE.splitlines(True),path)
    m=types.ModuleType('cf_S19');exec(compile(BASE,path,'exec'),m.__dict__);return m

def cf_namespace(host):
    g=host.__dict__;v=dict(g)
    for k,f in g.items():
        if isinstance(f,types.FunctionType) and f.__globals__ is g:
            h=types.FunctionType(f.__code__,v,f.__name__,f.__defaults__,f.__closure__);h.__kwdefaults__=f.__kwdefaults__;v[k]=h
    for k,value in CF_SOURCES.items():v[k]=dict(g[k],**value) if isinstance(value,dict) else value
    for k in ('r2__MODULES','wr__MODULES','_PP_MOD','_s16_global_C128_cs__MOD'):v[k]={}
    v['_ATTN_PORT_EXT']=None;v['_VQ8_MOD']=None
    # Pin S19's partition selection despite the oracle's changed register use.
    for fn in ('_attn_port_ext','partial_port_ext','_s16_global_C128_cs_module','r2_module','wr_module'):
        actual=v[fn];original=g[fn]
        def wrapped(*args,actual=actual,original=original):
            obj,old=actual(*args),original(*args)
            return types.SimpleNamespace(**{n:(getattr(old,n) if n in ('occupancy','max_cluster','sliding_resources') else getattr(obj,n)) for n in ('attn_forward','sliding_forward','occupancy','max_cluster','sliding_resources') if hasattr(obj,n)})
        v[fn]=wrapped
    def vq_module():
        if v['_VQ8_MOD'] is None:
            arch='90a' if torch.cuda.get_device_capability()[0]==9 else ''.join(map(str,torch.cuda.get_device_capability()))
            v['_VQ8_MOD']=load_inline(name='cf_vq_'+hashlib.sha256(v['VQ8_CUDA'].encode()).hexdigest()[:12]+'_'+arch,cpp_sources=v['VQ8_CPP'],cuda_sources=v['VQ8_CUDA'],functions=['vq8_forward','vq8_resources'],extra_cuda_cflags=['-O3',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
        return v['_VQ8_MOD']
    v['_vq8_module']=vq_module
    v['_cf_original_attention']=v['_best7_attention']
    v['_cf_original_fa3']=v['_attention_fa3_public']
    exec('''
def _attn_port_raw(qkv,data):
    return data['_cf_ready'].view(data['config']['batch'],data['config']['q_len'],-1)
def _best7_old_qk_prep(qkv,data,state,q_fp8=True):
    return data['_cf_fa3q'],data['_cf_gate']
def qk_prep(qkv,data,state,q_fp8=True):
    return data['_cf_fa3q'],data['_cf_gate']
def _attention_fa3_public(q,gate,data,state):
    return _cf_original_fa3(q,gate,dict(data,scheduler_metadata=data['_cf_metadata']),state)
def _best7_attention(qkv,data,state,plan):
    return _cf_original_attention((data['_cf_ready'].unsqueeze(0),1),data,state,plan)
''',v)
    return v

def cf_ready(host,state,d,existing=False):
    c=d['config'];B,L=c['batch'],c['q_len'];T=B*L;plan=host._s15n_plan(c,state)
    role='qkv3' if host._gate_split_on(d,state,plan) else 'qkv_gate'
    P,S=host._best7_project(d['attn_input'],role,state,plan)
    if P.dtype==torch.float32:
        raw=torch.zeros_like(P[0]);
        for i in range(S):raw=raw+P[i]
        raw=raw.bfloat16()
    else:raw=P.reshape(T,-1)
    q=raw[:,:1024].float().view(T,16,64);k=raw[:,1024:1152].float().view(T,2,64);val=raw[:,1152:1280].float().view(T,2,64)
    q=q*torch.rsqrt(q.square().mean(-1,keepdim=True)+1e-5);k=k*torch.rsqrt(k.square().mean(-1,keepdim=True)+1e-5)
    if c['window']:
        ang=d['positions'].float()[:,None]*state['inv_freq'][None,:];co=ang.cos()[:,None,:];sn=ang.sin()[:,None,:]
        def rope(x):
            a,b=x[...,0::2],x[...,1::2];return torch.stack((a*co-b*sn,a*sn+b*co),-1).flatten(-2)
        q,k=rope(q),rope(k)
    q8=q.to(torch.float8_e4m3fn);k8=(k/d['k_scale']).to(torch.float8_e4m3fn);v8=(val/d['v_scale']).to(torch.float8_e4m3fn)
    # Byte writes avoid unsupported float8 index_put on some image builds.
    for name,x in [('k_cache',k8),('v_cache',v8)]:d[name].view(torch.uint8).reshape(-1,2,64).index_copy_(0,d['cache_loc'],x.view(torch.uint8))
    half=(L==1 and not c['window']);ready=torch.zeros_like(raw);rb=ready.view(torch.uint8)
    qb=q.half().view(torch.uint8).reshape(T,-1) if half else q8.view(torch.uint8).reshape(T,-1)
    rb[:,:qb.shape[-1]]=qb;offset=2048 if half else 1024
    rb[:,offset:offset+128]=k8.view(torch.uint8).reshape(T,128);rb[:,offset+128:offset+256]=v8.view(torch.uint8).reshape(T,128)
    ready[:,1280:]=raw[:,1280:];gate=raw[:,1280:].float()
    # The FA3 scheduler is also setup between projection and the main. DEL
    # prepares its current-value metadata outside timing, without changing BASE.
    if not LOCAL and L==8 and ((B==16 and c['window']) or (B==1 and not c['window'])):
        from sgl_kernel.flash_attn import get_scheduler_metadata
        meta=get_scheduler_metadata(batch_size=B,max_seqlen_q=L,max_seqlen_k=d['page_table'].shape[1]*c['page_size'],num_heads=16,num_heads_k=2,headdim=64,cache_seqlens=d['cache_seqlens'],qkv_dtype=q8.dtype,cu_seqlens_q=d['cu_seqlens_q'],page_size=c['page_size'],causal=True,window_size=(c['window']-1,0) if c['window'] else (-1,-1),num_splits=0,pack_gqa=None)
        if existing:d['_cf_metadata'].copy_(meta)
        else:d['_cf_metadata']=meta
    if existing:
        d['_cf_ready'].copy_(ready);d['_cf_fa3q'].copy_(q8);d['_cf_gate'].copy_(gate)
    else:d.update(_cf_ready=ready,_cf_fa3q=q8,_cf_gate=gate)
    return dict(role=role,splits=S,dtype=str(P.dtype),width=P.shape[-1],half_Q=half)

def cf_copy(dst,src):
    for k,value in src.items():
        if isinstance(value,torch.Tensor):dst[k].copy_(value)
        elif isinstance(value,dict):cf_copy(dst[k],value)

def cf_graph(fn):
    for _ in range(3):out=fn()
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn()
    return graph,out

def cf_time(graphs,seed_index):
    names=list(graphs);times={n:[] for n in names};enq={n:[] for n in names};orders=list(s21_orders(names,ROUNDS,seed_index));telemetry=[s21_telemetry()]
    start_ms=telemetry[0].get('t_s',0)*1000
    for ri,order in enumerate(orders):
        evs=[]
        for n in order:
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            lab_cold_l2();t0=time.perf_counter_ns();a.record();graphs[n][0].replay();b.record();enq[n].append(round((time.perf_counter_ns()-t0)/1000));evs.append((n,a,b))
        torch.cuda.synchronize()
        for n,a,b in evs:times[n].append(round(a.elapsed_time(b)*1e6))
        if ri%2:telemetry.append(s21_telemetry())
    compact=[[round(v.get('t_s',0)*1000-start_ms),v.get('sm_MHz',0),v.get('mem_MHz',0),v.get('temperature_C',0),v.get('power_mW',0)] for v in telemetry]
    return times,enq,compact

def lab(deadline,weights,state=None):
    started=time.monotonic();host=cf_host()
    if globals().get('CF_LOCAL_INSTALL'):CF_LOCAL_INSTALL(host)
    weights=PUBLIC_SPEC['generate_weights'](74000033,device='cuda',router_seed=74000039) if weights is None else weights
    state=host.prepare(weights);v=cf_namespace(host)
    if globals().get('CF_LOCAL_NAMESPACE'):CF_LOCAL_NAMESPACE(host,v)
    from concurrent.futures import ThreadPoolExecutor
    if not LOCAL:
      with ThreadPoolExecutor(max_workers=6) as pool:
        fns=[lambda:v['_s16_global_C128_cs_module'](),lambda:v['r2_module']('dsm'),lambda:v['wr_module']('STREAM2'),lambda:v['partial_port_ext'](False),lambda:v['_attn_port_ext'](),lambda:v['_vq8_module']()]
        for f in list(pool.map(lambda f:f(),fns)):pass
    act=tl_activity();rows=[];top=[];checks=dict(BASE=0,AA=0,DEL_fixture_pass=0,DEL_fixture_fail=0);profiles={}
    for B,L in CASES:
      for W in (2048,0):
        label=('d' if L==1 else 'v')+str(B)+('S' if W else 'F')
        def make(seed):return host._ap_generate_input(batch=B,q_len=L,ctx=80000,layer='sliding' if W else 'full',spread='fixed' if B==1 else 'agentic',dist='soft' if W else 'coherent',page=1,seed=seed,device='cuda')
        data=make(SEEDS[0]);abi=cf_ready(host,state,data);graphs={}
        for n in (('A','A2','DEL') if (B+L+W)%2 else ('DEL','A2','A')):
            call=(lambda:host.kernel_fn(data,state)) if n!='DEL' else (lambda:v['kernel_fn'](data,state))
            graphs[n]=cf_graph(call)
        graphs={n:graphs[n] for n in ('A','A2','DEL')};samples=[];tel=[];fixture=[]
        for si,seed in enumerate(SEEDS):
            if time.monotonic()>deadline-12:raise TimeoutError(('incomplete oracle',label,si))
            if si:cf_copy(data,make(seed));cf_ready(host,state,data,True)
            checker=eh_checker(data,weights)
            for n in ('A','A2'):
                graphs[n][0].replay();torch.cuda.synchronize();ok,msg=checker(graphs[n][1]);assert ok,(label,seed,n,msg);checks['BASE' if n=='A' else 'AA']+=1
            graphs['DEL'][0].replay();torch.cuda.synchronize();ok,msg=checker(graphs['DEL'][1]);fixture.append(bool(ok));checks['DEL_fixture_pass' if ok else 'DEL_fixture_fail']+=1
            del checker
            ns,enq,t=cf_time(graphs,si);samples.append([ns[n] for n in graphs]);tel.append(t)
            severity=max(max(ns[n])/statistics.median(ns[n]) for n in graphs)
            top.append(dict(label=label,seed=seed,severity=severity,enqueue_us=[enq[n] for n in graphs]));top.sort(key=lambda x:x['severity'],reverse=True);del top[4:]
            if si==0:
                profiles[label]={}
                for n in ('A','DEL'):
                    lab_cold_l2();torch.cuda.synchronize();act.profile_begin();graphs[n][0].replay();torch.cuda.synchronize();nodes=act.profile_end();origin=min(x[0] for _,x in nodes)
                    profiles[label][n]=dict(total_nodes=len(nodes),attention=[[name[:48],x[0]-origin,x[1]-x[0],list(x[7:13])] for name,x in nodes if re.search(r'(sliding_|attn_main|attn_split|attn_combine|vq8_|qk_prep|sumcast|flash_|prepare_varlen|norm_gate)',name)])
        rows.append(dict(label=label,B=B,L=L,window=W,round_ns=samples,telemetry=tel,DEL_fixture_TASK=fixture,ABI=abi))
        graphs=data=None;gc.collect();torch.cuda.empty_cache()
    packet=dict(kind='P5_FINISH_DELETION_ORACLE',scope='Excluded-work oracle only; ready Q/K/V/gate prepared once per fresh fixture outside timing, including current FA3 scheduler metadata. Never a candidate; TASK is not a validity gate for DEL. Producer, all-key attention/math/combine/downstream stages remain timed.',base_sha256=hashlib.sha256(BASE.encode()).hexdigest(),source_hashes=CF_HASHES,identities=dict(base_sha256=hashlib.sha256(BASE.encode()).hexdigest(),sources=CF_HASHES),seeds=list(SEEDS),seed_ids=list(SEEDS),measurement_protocol=dict(name='palindromic-cohort-v1',rounds=ROUNDS,seed_count=len(SEEDS),aggregation='arithmetic-mean-all-rounds',selective_retiming=False),arms=['A','A2','DEL'],checks=checks,paired=rows,telemetry_columns=['relative_ms','sm_MHz','mem_MHz','temperature_C','power_mW'],telemetry_error=S21_TELEMETRY_ERROR,worst_enqueue_traces=top,profiles=profiles,attention_selections=dict(r2=str(state.get('r2_rows')),wr=str(state.get('wr_rows')),vq8=str(state.get('vq8_rows')),partial=str(state.get('partial_port_rows'))),complete=True,elapsed_s=time.monotonic()-started)
    if globals().get('CF_LOCAL_ARCHIVE'):CF_LOCAL_ARCHIVE(packet)
    packet=pack_samples(packet)
    return dict(packet=packet,logs=['P5 consumer finish-deletion oracle complete; DEL is excluded-work, not a performance candidate.'])
