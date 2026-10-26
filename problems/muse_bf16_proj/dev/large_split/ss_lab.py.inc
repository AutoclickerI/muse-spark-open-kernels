def ss_load():
    path='<P10 SS>';linecache.cache[path]=(len(SS_BASE),None,SS_BASE.splitlines(True),path);m=types.ModuleType('ls_ss');m.__file__=path;exec(compile(SS_BASE,path,'exec'),m.__dict__)
    if globals().get('SS_LOCAL_INSTALL'):SS_LOCAL_INSTALL(m)
    return m.__dict__

def lab(deadline,weights,state=None):
    import contextlib,io,gc,traceback
    start=time.monotonic();m=ss_load();task={};check={}
    for i,s in enumerate(SS_TASK):exec(compile(s,'<published layer '+str(i)+'>','exec'),task)
    exec(compile(SS_CHECK,'<published projection TASK>','exec'),check)
    weights=task['generate_weights'](16000311,router_seed=16000313);log=io.StringIO();t0=time.monotonic()
    with contextlib.redirect_stdout(log):st=m['prepare'](weights)
    base_prepare=time.monotonic()-t0;bank=m['ss_prepare'](st);rows=[];failure=None;seeds=[16000319+104729*i for i in range(8)]
    def capture(fn):
        fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):out=fn()
        g.replay();torch.cuda.synchronize();return g,out
    def times(gs):
        v=[[] for g in gs]
        for r in range(8):
            events=[]
            for j in (range(len(gs)) if r%2==0 else range(len(gs)-1,-1,-1)):
                lab_cold_l2();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();gs[j].replay();b.record();events.append((j,a,b))
            torch.cuda.synchronize()
            for j,a,b in events:v[j].append(a.elapsed_time(b)*1000)
        return [statistics.median(a) for a in v]
    empty=times([capture(lambda:None)[0]])[0]
    try:
        for t in (128,256):
            for role in ('qkv_gate','o','pool_router','unpooler'):
                if time.monotonic()>deadline-25:raise TimeoutError('before '+str((t,role)))
                w=st[role];n,k=w.shape;x=torch.randn(t,k,device='cuda',dtype=torch.bfloat16);res=torch.randn(t,8192,device='cuda');plan=st['best7_plans'][t//8,8,2048];data=dict(local_start=torch.zeros((),device='cuda',dtype=torch.int32))
                arms=[('A',),('AA',)]+[(i,s) for i,c in enumerate(m['SS_CONFIGS']) if c[0]==t for s in (1,2,4,8)];labels=[list(a) for a in arms]
                def run(arm):
                    if len(arm)==1:return m['_best7_project'](x,role,st,plan,torch.float32 if role=='pool_router' else torch.bfloat16)
                    return m['ss_project'](x,w,bank,*arm)
                def consume(pair):return m['_best7_route'](pair,st,data,plan) if role=='pool_router' else m['_best7_boundary'](pair,res,st,plan,'b1' if role=='o' else 'b2')
                graphs=[capture(lambda arm=a:run(arm)) for a in arms];client=role!='qkv_gate'
                if client:graphs += [capture(lambda arm=a:consume(run(arm))) for a in arms]
                vals=[[] for g in graphs];errors=[0.]*len(arms);router_error=[0.]*len(arms)
                for seed in seeds:
                    gen=torch.Generator(device='cuda').manual_seed(seed);x.copy_(torch.randn(x.shape,device='cuda',generator=gen).bfloat16());res.copy_(torch.randn(res.shape,device='cuda',generator=gen));d=dict(x=x,config=dict(layer='qkv_gate',m=t,n=n,k=k));wt=dict(qkv_gate=w)
                    for j,arm in enumerate(arms):
                        pair=run(arm);v=pair[0].sum(0);out=v.bfloat16();ok,msg=check['check_implementation'](d,out,wt);assert ok,(t,role,arm,seed,msg);errors[j]=max(errors[j],check['relative_l2'](out,check['ref_kernel'](d,wt)))
                        if role=='pool_router':
                            r=v[:,4096:4352]+v[:,4352:];rd=dict(x=x,config=dict(layer='router',m=t,n=256,k=k));ok,msg=check['check_implementation'](rd,r,weights);assert ok,(t,arm,seed,'FP32 router',msg);router_error[j]=max(router_error[j],check['relative_l2'](r,check['ref_kernel'](rd,weights)))
                    for j,v in enumerate(times([g for g,o in graphs])):vals[j].append(round((v-empty)*125))
                    x.zero_();x[:,seed%k]=16
                    for arm in arms:
                        pair=run(arm);v=pair[0].sum(0);ok,msg=check['check_implementation'](d,v.bfloat16(),wt);assert ok,(t,role,arm,'column',msg)
                        if role=='pool_router':
                            r=v[:,4096:4352]+v[:,4352:];rd=dict(x=x,config=dict(layer='router',m=t,n=256,k=k));ok,msg=check['check_implementation'](rd,r,weights);assert ok,(t,arm,'FP32 router column',msg)
                rows.append(dict(t=t,role=role,arms=labels,producer=vals[:len(arms)],neighbour=vals[len(arms):] if client else None,errors=errors,router_errors=router_error));graphs=None;gc.collect();torch.cuda.empty_cache()
    except Exception as e:failure=repr(e)+' | '+' | '.join(traceback.format_exc().splitlines()[-6:])
    result=dict(rows=rows,unit_us=.008,seeds=seeds,complete=len(rows)==8 and failure is None,failure=failure,base_prepare_s=base_prepare,added_prepare_s=bank['prepare_s'],resources=bank['resources'],elapsed_s=time.monotonic()-start)
    if globals().get('SS_LOCAL_ARCHIVE'):SS_LOCAL_ARCHIVE(result)
    blob=base64.b85encode(lzma.compress(json.dumps(result,separators=(',',':')).encode())).decode();assert len(blob)<8300,len(blob)
    return ['META P10 shared-weight token-WG SS:all8large sites24arms8seeds8cold S19+AA originalFP32routerTASK real neighbours','SS_XZ85 '+blob,'END complete='+str(result['complete'])]
