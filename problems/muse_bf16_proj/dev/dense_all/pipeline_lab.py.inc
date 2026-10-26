def pipe_load():
    path='<P10 pipeline>';linecache.cache[path]=(len(PIPE_SOURCE),None,PIPE_SOURCE.splitlines(True),path);ns={'__name__':'da_pipeline'};exec(compile(PIPE_SOURCE,path,'exec'),ns)
    if globals().get('PIPE_LOCAL_INSTALL'):PIPE_LOCAL_INSTALL(ns)
    return ns

def lab(deadline,weights,state=None):
    import contextlib,io,gc,traceback
    start=time.monotonic();m=pipe_load();task={};check={}
    for i,s in enumerate(PIPE_TASK):exec(compile(s,'<published layer '+str(i)+'>','exec'),task)
    exec(compile(PIPE_CHECK,'<published projection TASK>','exec'),check)
    weights=task['generate_weights'](13000311,router_seed=13000313);log=io.StringIO()
    with contextlib.redirect_stdout(log):st=m['prepare'](weights)
    bank=m['da_prepare'](weights);ns=bank['ns'];instances=[(bn,64,1,2,2,coded,2) for bn in (64,128,256) for coded in (False,True)];modules={};compile_s={}
    for piped in (False,True):
        src=m['da_pipeline_source'](ns,instances,piped);t0=time.monotonic();tag=hashlib.sha256(src.encode()).hexdigest()[:12]
        modules[piped]=None if globals().get('PIPE_LOCAL_INSTALL') else m['load_inline'](name='p10_pipeline_'+tag,cpp_sources=ns['dd_decodediet_tt_mw_CPP'],cuda_sources=src,functions=['launch','info'],extra_cuda_cflags=['-O3','-std=c++20',ns['_nf_arch_flag']()],extra_ldflags=['-lcuda'],no_implicit_headers=True,with_pytorch_error_handling=False);compile_s[str(piped)]=time.monotonic()-t0
    raw={}
    for role in ('qkv_gate','o','pool_router','unpooler'):
        w=bank['roles'][role];p=ns['dd_decodediet_tt_mw_pack_weight'](w,64,1,1,False);raw[role]=p['w']
    def capture(fn):
        fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):out=fn()
        g.replay();torch.cuda.synchronize();return g,out
    def times(gs):
        out=[[] for _ in gs]
        for r in range(8):
            events=[]
            for j in (range(len(gs)) if r%2==0 else range(len(gs)-1,-1,-1)):
                lab_cold_l2();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();gs[j].replay();b.record();events.append((j,a,b))
            torch.cuda.synchronize()
            for j,a,b in events:out[j].append(a.elapsed_time(b)*1000)
        return [statistics.median(v) for v in out]
    seeds=[13000319+104729*i for i in range(8)];rows=[];failure=None
    try:
        for t in (128,256):
            plan=st['best7_plans'][t//8,8,2048]
            for role in ('qkv_gate','o','pool_router','unpooler'):
                if time.monotonic()>deadline-20:raise TimeoutError(str((t,role)))
                w=bank['roles'][role];n,k=w.shape;x=torch.randn(t,k,device='cuda',dtype=torch.bfloat16);res=torch.randn(t,8192,device='cuda');data=dict(local_start=torch.zeros((),device='cuda',dtype=torch.int32))
                # Factorial: identical raw/coded RS math, with/without overlap,
                # full/half token tiles and every legal disjoint K split.
                arms=[('A',),('AA',)]+[(p,c,bn,s) for p in (False,True) for c in (True,False) for bn in (t,t//2) for s in (2,4,8)]
                def run(arm):
                    if len(arm)==1:return m['_best7_project'](x,role,st,plan,torch.float32 if role=='pool_router' else torch.bfloat16)
                    piped,coded,bn,s=arm;p=next(v for (tt,rr,a),v in bank['wide'].items() if tt==t and rr==role and v['splits']==s);out=torch.empty((s,t,n),device='cuda');wp=p['wp'] if coded else raw[role]
                    if globals().get('PIPE_LOCAL_INSTALL'):out.zero_();m['DA_LOCAL_MM'](x,w,out[0])
                    else:
                        idx=instances.index((bn,64,1,2,2,coded,2));rc=modules[piped].launch(idx,x.data_ptr(),wp.data_ptr(),p['esc'].data_ptr(),out.data_ptr(),t,n,k,s,0,torch.cuda.current_stream().cuda_stream)
                        if rc:raise RuntimeError('pipeline native '+str((t,role,arm,rc)))
                    return out,s
                def consume(pair):
                    if role=='pool_router':return m['_best7_route'](pair,st,data,plan)
                    if role=='qkv_gate':return pair
                    return m['_best7_boundary'](pair,res,st,plan,'b1' if role=='o' else 'b2')
                graphs=[capture(lambda arm=a:run(arm)) for a in arms];client=role!='qkv_gate'
                if client:graphs += [capture(lambda arm=a:consume(run(arm))) for a in arms]
                values=[[] for _ in graphs];errors=[0.]*len(arms)
                for seed in seeds:
                    gen=torch.Generator(device='cuda').manual_seed(seed);x.copy_(torch.randn(x.shape,device='cuda',generator=gen).bfloat16());res.copy_(torch.randn(res.shape,device='cuda',generator=gen));d=dict(x=x,config=dict(layer='qkv_gate',m=t,n=n,k=k));wt=dict(qkv_gate=w)
                    for j,arm in enumerate(arms):
                        out=m['da_materialize'](run(arm));ok,msg=check['check_implementation'](d,out,wt);assert ok,(t,role,arm,seed,msg)
                        errors[j]=max(errors[j],check['relative_l2'](out,check['ref_kernel'](d,wt)))
                    for j,v in enumerate(times([g for g,o in graphs])):values[j].append(round(v*125))
                    old=x.clone();x.zero_();x[:,seed%k]=16
                    for arm in arms:
                        ok,msg=check['check_implementation'](d,m['da_materialize'](run(arm)),wt);assert ok,(t,role,arm,'column',msg)
                    x.copy_(old)
                rows.append(dict(t=t,role=role,arms=arms,producer=values[:len(arms)],neighbour=values[len(arms):] if client else None,errors=errors))
                graphs=None;gc.collect();torch.cuda.empty_cache()
    except Exception as error:failure=repr(error)+' | '+' | '.join(traceback.format_exc().splitlines()[-5:])
    archive=dict(unit_us=.008,rows=rows,failure=failure,seeds=seeds,complete=len(rows)==8 and failure is None,elapsed_s=time.monotonic()-start,compile_s=compile_s)
    if globals().get('PIPE_LOCAL_ARCHIVE'):PIPE_LOCAL_ARCHIVE(archive)
    blob=base64.b85encode(lzma.compress(json.dumps(archive,separators=(',',':')).encode())).decode();assert len(blob)<8300,len(blob)
    return ['META P10 register-A decoder/MMA overlap factorial,raw exact oracle,8seeds8cold,actual neighbours','PIPE_XZ85 '+blob,'END complete='+str(archive['complete'])]
