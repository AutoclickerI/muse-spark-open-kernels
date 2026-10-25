def lab(deadline,weights):
    import linecache,concurrent.futures,ctypes
    started=time.monotonic();weights,kind=cl_weights(weights)
    def module(label,source):
        file='<nf_port_lab_'+label+'>';linecache.cache[file]=(len(source),None,source.splitlines(True),file)
        ns={'__name__':label,'__file__':file};exec(compile(source,file,'exec'),ns);return ns
    port=module('port',PORT_SOURCE);old=module('S14',S13_SOURCE)
    hi=weights['router'].to(torch.bfloat16)
    state={'qkv_gate':weights['qkv_gate'],'qkv3':weights['qkv_gate'][:1280].contiguous(),'o':weights['o'],
        'pool_router':torch.cat([weights['pooler'],hi,(weights['router']-hi.float()).to(torch.bfloat16)]),
        'unpooler':weights['unpooler'],'unpooler_slice':weights['unpooler'][:1024].contiguous()}
    native=torch.cuda.get_device_capability()==(9,0)
    def prepare_old():
        old['_seed_v9_gemv_choices'](state);old['_gate_split_prepare'](state,torch.device('cuda'))
        old['cd_module'](((1,1,2,3),));old['_cd_prepare'](state)
        state['proj_plan']=dict(old['PROJ_PINS']);state['best7_stream_ops']={}
        state['best7_stream_ops'][1,'o',6,1]=old['Best7ProjectionCell'](state['o'],1,6,fp32=False)
        if native:
            state['lt_mod']=old['_lt_module']();state['lt_workspace']=torch.empty(32<<20,device='cuda',dtype=torch.uint8);state['lt_algos']={}
            for t,name in ((128,'o'),(256,'qkv_gate')):
                w=state[name];x=torch.zeros((t,w.shape[1]),device='cuda',dtype=torch.bfloat16);out=torch.empty((t,w.shape[0]),device='cuda',dtype=torch.bfloat16)
                state['lt_algos'][name,t]=state['lt_mod'].lt_options(w,x,out)
            # Exact S13 short-LT recipes for full unpooler, not a generic torch.mm control.
            src=old['PG_RAW'];tag=hashlib.sha256(src.encode()).hexdigest()[:12]
            path=old['load_inline'](name=f'muse_decode_measured_lt_{tag}_90a',cpp_sources='',cuda_sources=src,functions=None,is_python_module=False,
                no_implicit_headers=True,extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a'],extra_ldflags=['-lcublasLt'],verbose=False)
            bank={'mod':old['_PgNative'](path),'layouts':{},'workspace':torch.empty(4<<30,device='cuda',dtype=torch.uint8)}
            for t,nn in ((128,1),(256,0)):
                w=state['unpooler'];codes,_=bank['mod'].options(t,w.shape[0],w.shape[1],nn,False,bank['workspace'].numel())
                bank['layouts'][t,'unpooler',nn]=(w.T.contiguous() if nn else w,codes)
            state['measured_lt']=bank
        return state
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        future=pool.submit(prepare_old);ps=port['nf_prepare'](weights);future.result()
    dev=weights['o'].device
    for key,size in (('alpha1',8192),('alpha2',8192),('beta1',8192),('beta2',8192),('pre_ffn_w',8192),('next_pre_attn_w',8192),('post_ffn_w',8192),('pre_expert_w',4096)):
        state[key]=torch.full((size,),.5 if key.startswith(('alpha','beta')) else 1.,device=dev)
    state['router_bias']=torch.zeros(256,device=dev)
    state['inv_freq']=(1./(500000.**(torch.arange(0,64,2,device=dev).float()/64))).contiguous()
    empty=statistics.median(_cl_times([_cl_graph(lambda:None)],8)[0])
    lines=[f'META NF attributed port vs exact S14 f1c77ce6 projection dispatch; {kind}; native={native}; 8 seeds x12 cold ABBA; setup={time.monotonic()-started:.1f}s empty={empty:.2f}',
        'META FP32 fused pool_router, same consumers; qkv3 uses materialize proxy (whole-layer gate required); t1 coded stays ours unless paired whole layer authorizes change']
    for t,name in sorted(port['NF_SMALL'].keys()|port['NF_WIDE']):
        if time.monotonic()>deadline-8:lines.append(f'SKIP {t} {name}');continue
        try:
            w=state[name];x=torch.randn((t,w.shape[1]),device=dev,dtype=torch.bfloat16)
            stream=torch.randn((t,8192),device=dev,dtype=torch.float32)
            data={'local_start':torch.zeros((),device=dev,dtype=torch.int32),'positions':torch.arange(t,device=dev,dtype=torch.int64),'cache_loc':torch.arange(t,device=dev,dtype=torch.int64),
              'k_cache':torch.empty((t,1,2,64),device=dev,dtype=torch.float8_e4m3fn),'v_cache':torch.empty((t,1,2,64),device=dev,dtype=torch.float8_e4m3fn),
              'k_scale':torch.ones((),device=dev),'v_scale':torch.ones((),device=dev),'config':dict(tokens=t,rope_theta=500000.,qk_scale_factor=5.4730064863875)}
            def baseline():
                if not native and t>32:return (torch.mm(x,w.T,out_dtype=torch.float32).unsqueeze(0),1)
                if t==1 and name=='o':return state['best7_stream_ops'][1,'o',6,1](x).unsqueeze(0),1
                variant=old['PROJ_PINS'].get((t,name),'custom')
                return old['_project_variant'](x,w,name,state,variant,torch.float32 if name=='pool_router' else torch.bfloat16)
            def consume(pair):
                if name=='qkv3':return old['_materialize'](pair).to(torch.bfloat16)
                if name=='qkv_gate':return old['qk_prep'](pair,data,state,True)
                if name=='pool_router':
                    if native and t<=32:return old['_glue_cl_route'](pair,state,data,('cl',8,256,0))
                    return old['route_and_norm'](pair,state,data)
                post=name.startswith('unpooler')
                args=(pair,stream,state['alpha2' if post else 'alpha1'],state['beta2' if post else 'beta1'],state['next_pre_attn_w' if post else 'pre_ffn_w'])
                if native and t<=32:return old['_glue_cl_boundary'](*args,state['post_ffn_w'] if post else None,('cl',8,256,1))
                return old['boundary'](*args,postw=state['post_ffn_w'] if post else None)
            a=baseline();b=port['nf_project'](x,name,ps)
            ref=x.double()@w.double().T
            e=float((b[0].double().sum(0)-ref).norm()/ref.norm());tol=.004 if b[0].dtype==torch.bfloat16 else 3e-5
            if e>tol:raise RuntimeError('projection error '+str(e))
            graphs=[_cl_graph(lambda:consume(baseline())),_cl_graph(lambda:consume(baseline())),_cl_graph(lambda:consume(port['nf_project'](x,name,ps)))]
            if t==32 and name=='pool_router':
                graphs.append(_cl_graph(lambda:consume((gb_run(x,w,64,6,4),6))))
            ratios=[];aa=[];au=[];bu=[];ns4=[];probe=0.
            for seed in (61791,14357,92003,17453,51613,72577,35111,81901):
                g=torch.Generator(device=dev).manual_seed(seed);x.copy_(torch.randn(x.shape,device=dev,generator=g).to(torch.bfloat16))
                x[-1].zero_();x[-1,(seed*131+t)%x.shape[1]]=16.
                pp=port['nf_project'](x,name,ps)[0].double().sum(0);rr=x.double()@w.double().T
                er=float((pp-rr).norm()/rr.norm());e=max(e,er)
                probe=max(probe,float((pp[-1]-rr[-1]).norm()/rr[-1].norm()))
                if probe>1e-5:raise RuntimeError('one-hot probe '+str(probe))
                if er>tol:raise RuntimeError('seed error '+str(er))
                vals=_cl_times(graphs,12)
                ratios.append(statistics.median((b-empty)/(a-empty) for a,b in zip(vals[0],vals[2])))
                aa.append(statistics.median((b-empty)/(a-empty) for a,b in zip(vals[0],vals[1])))
                au.append(statistics.median(vals[0])-empty);bu.append(statistics.median(vals[2])-empty)
                if len(graphs)>3:ns4.append(statistics.median((v-empty)/(a-empty) for a,v in zip(vals[0],vals[3])))
            lines.append(f'R t={t} {name} A={statistics.mean(au):.2f} B={statistics.mean(bu):.2f} r={statistics.mean(ratios):.5f}[{min(ratios):.5f},{max(ratios):.5f}] aa={max(abs(v-1) for v in aa):.5f} e={e:.1e} p={probe:.1e}'+(f' NS4/bal2={statistics.mean(ns4):.5f}' if ns4 else ''))
            del graphs
        except Exception as ex:
            lines.append(f'ERROR {t} {name} {type(ex).__name__}: {str(ex)[:240]}')
            # A launch fault is not recoverable by testing another row.
            torch.cuda.synchronize()
    return lines+[f'END elapsed={time.monotonic()-started:.1f}s']
