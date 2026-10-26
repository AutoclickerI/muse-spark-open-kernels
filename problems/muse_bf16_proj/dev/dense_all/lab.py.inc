def da_load():
    import linecache
    path='<P10 dense-all>';linecache.cache[path]=(len(DA_SOURCE),None,DA_SOURCE.splitlines(True),path);ns={'__name__':'dense_all'};exec(compile(DA_SOURCE,path,'exec'),ns);return ns

def lab(deadline,weights,state=None):
    import time,statistics,math,json,lzma,base64,struct
    begin=time.monotonic();m=da_load();T=torch;task={}
    for i,source in enumerate(DA_TASK_SOURCES):exec(compile(source,'<published layer TASK '+str(i)+'>','exec'),task)
    checker={};exec(compile(DA_CHECK_SOURCE,'<published projection TASK>','exec'),checker)
    weights=task['generate_weights'](910031,router_seed=910039)
    T.cuda.synchronize();a=time.monotonic();st=m['prepare'](weights);T.cuda.synchronize();base_prepare=time.monotonic()-a
    bank=m['da_prepare'](weights);records=[];seeds=(137203,282089,409891,563753,697181,811117,947239,1083709)
    preflight_begin=time.monotonic()
    # Warm every native entry point before any graph capture. An invalid launch
    # must identify its shape directly rather than poison the later host setup.
    for pt in (1,8,16,32,128,256):
        for pr in ('qkv_gate','qkv3','gate','o','pool_router','unpooler'):
            actual='unpooler_slice' if pr=='unpooler' and pt<=8 else pr;w=bank['roles'][actual]
            xx=T.zeros((pt,w.shape[1]),device='cuda',dtype=T.bfloat16)
            for arm in range(len(m['DA_SMALL'] if pt<=32 else m['DA_WIDE'])):
                try:
                    pair=m['da_project'](xx,pr,bank,arm);T.cuda.synchronize()
                    if pr=='gate' and pt<=32:m['da_project'](xx,pr,bank,arm,T.ones((pt,1024),device='cuda',dtype=T.bfloat16));T.cuda.synchronize()
                except Exception as error:raise RuntimeError('native preflight '+str((pt,pr,arm))+': '+repr(error)) from error
    preflight_s=time.monotonic()-preflight_begin
    def capture(fn):
        fn();T.cuda.synchronize();g=T.cuda.CUDAGraph()
        with T.cuda.graph(g):out=fn()
        g.replay();T.cuda.synchronize();return g,out
    def times(gs):
        ev=[]
        for r in range(8):
            for j in (range(len(gs)) if r%2==0 else range(len(gs)-1,-1,-1)):
                lab_cold_l2();a,b=T.cuda.Event(enable_timing=True),T.cuda.Event(enable_timing=True);a.record();gs[j].replay();b.record();ev.append((j,a,b))
                if globals().get('DA_LOCAL_SYNC'):
                    try:T.cuda.synchronize()
                    except Exception:
                        print('DA_TIMING_FAILURE',j,r,flush=True);raise
        T.cuda.synchronize();v=[[] for _ in gs]
        for j,a,b in ev:v[j].append(a.elapsed_time(b)*1000)
        return [statistics.median(x) for x in v]
    empty=times([capture(lambda:None)[0]])[0]
    def row(t,role,kind,labels,vals,extra=None):
        samples=[[round(v*125) for v in values] for values in vals]
        records.append(dict(t=t,role=role,kind=kind,arms=labels,values=samples,**(extra or {})))
    failure=None;phase='start'
    try:
        for t in (1,8,16,32,128,256):
            batch,q=(t,1) if t in (1,16,32) else (t//8,8)
            plan=st['best7_plans'][batch,q,2048]
            for role in ('qkv_gate','qkv3','gate','o','pool_router','unpooler'):
                if globals().get('DA_LOCAL_CASES') and (t,role) not in DA_LOCAL_CASES:continue
                if globals().get('DA_LOCAL_TRACE'):print('DA_CELL',t,role,flush=True)
                if time.monotonic()>deadline-30:raise TimeoutError('dense-all budget before '+str((t,role)))
                actual='unpooler_slice' if role=='unpooler' and t<=8 else role;w=bank['roles'][actual];n,k=w.shape
                x=T.randn((t,k),device='cuda',dtype=T.bfloat16);res=T.randn((t,8192),device='cuda');st[actual]=w
                data=dict(local_start=T.zeros((),device='cuda',dtype=T.int32))
                arms=[-1,-1]+list(range(len(m['DA_SMALL'] if t<=32 else m['DA_WIDE'])));labels=['A','AA']+list(range(len(arms)-2))
                def producer(arm):
                    if arm>=0:return m['da_project'](x,role,bank,arm)
                    if role=='gate' or role=='qkv3' and t not in (1,16,32):return T.mm(x,w.T,out_dtype=T.float32).unsqueeze(0),1
                    return m['_best7_project'](x,actual,st,plan,T.float32 if role=='pool_router' else T.bfloat16)
                def consume(pair):
                    if role=='pool_router':return m['_best7_route'](pair,st,data,plan)
                    return m['_best7_boundary'](pair,res,st,plan,'b1' if role=='o' else 'b2')
                graphs=[];client=role in ('o','pool_router','unpooler')
                for arm in arms:
                    phase='capture producer '+str(arm);graphs.append(capture(lambda arm=arm:producer(arm)))
                if client:
                    for arm in arms:
                        phase='capture neighbour '+str(arm);graphs.append(capture(lambda arm=arm:consume(producer(arm))))
                vals=[[] for _ in graphs];errors=[0.]*len(arms);probes=[0.]*len(arms)
                for seed in seeds:
                    gen=T.Generator(device='cuda').manual_seed(seed);x.copy_(T.randn(x.shape,device='cuda',generator=gen).bfloat16());res.copy_(T.randn(res.shape,device='cuda',generator=gen))
                    check_data=dict(x=x,config=dict(layer='qkv_gate',m=t,n=n,k=k));check_weights=dict(qkv_gate=w)
                    for j,arm in enumerate(arms):
                        phase='random TASK '+str((seed,arm))
                        pair=producer(arm);out=m['da_materialize'](pair);ok,msg=checker['check_implementation'](check_data,out,check_weights);assert ok,(t,role,j,seed,msg)
                        if role=='pool_router':
                            raw=pair[0].sum(0).float();router=raw[:,4096:4352]+raw[:,4352:];rd=dict(x=x,config=dict(layer='router',m=t,n=256,k=k));ok,msg=checker['check_implementation'](rd,router,weights);assert ok,(t,role,j,seed,'FP32 router',msg)
                        errors[j]=max(errors[j],checker['relative_l2'](out,checker['ref_kernel'](check_data,check_weights)))
                    phase='timing '+str(seed)
                    for j,v in enumerate(times([g for g,_ in graphs])):vals[j].append(v-empty)
                    original=x.clone();x.zero_();x[:,(seed*131)%k]=16
                    for j,arm in enumerate(arms):
                        pair=producer(arm);out=m['da_materialize'](pair);ok,msg=checker['check_implementation'](check_data,out,check_weights);assert ok,(t,role,j,seed,'column',msg)
                        if role=='pool_router':
                            raw=pair[0].sum(0).float();router=raw[:,4096:4352]+raw[:,4352:];rd=dict(x=x,config=dict(layer='router',m=t,n=256,k=k));ok,msg=checker['check_implementation'](rd,router,weights);assert ok,(t,role,j,seed,'FP32 router column',msg)
                        probes[j]=max(probes[j],checker['relative_l2'](out,checker['ref_kernel'](check_data,check_weights)))
                    x.copy_(original)
                byte_counts=[];issued=[]
                for arm in range(len(arms)-2):
                    p=bank['small' if t<=32 else 'wide'][t,actual,arm]
                    byte_counts.append(p['wp'].numel()*p['wp'].element_size()+p['esc'].numel()*p['esc'].element_size())
                    if t<=32:issued.append(byte_counts[-1])
                    else:
                        bn=m['DA_WIDE'][arm][0] or t;issued.append((p['wp'].numel()*p['wp'].element_size()+p['metadata_requests'])*((t+bn-1)//bn))
                row(t,role,'producer',labels,vals[:len(arms)],dict(errors=errors,probes=probes,code_bytes=byte_counts,issued_weight_byte_bound=issued,control='diagnostic torch FP32-output matmul; this projection is inactive' if role=='gate' or role=='qkv3' and t not in (1,16,32) else 'S18'))
                if client:row(t,role,'neighbour',labels,vals[len(arms):])
                if role=='gate' and t<=32:
                    y=T.randn((t,1024),device='cuda',dtype=T.bfloat16)
                    def apply(arm):return m['_gate_split_apply'](x,y,st) if arm<0 else m['da_project'](x,'gate',bank,arm,y)
                    gs=[capture(lambda arm=arm:apply(arm)) for arm in arms];vs=[[] for _ in gs];diagnostics=[]
                    for seed in seeds:
                        gen=T.Generator(device='cuda').manual_seed(seed);x.copy_(T.randn(x.shape,device='cuda',generator=gen).bfloat16());y.copy_(T.randn(y.shape,device='cuda',generator=gen).bfloat16());ref=(y.float()*T.sigmoid((x.double()@w.double().T).float())).bfloat16()
                        for g,out in gs:g.replay();T.cuda.synchronize();diagnostics.append(checker['relative_l2'](out,ref))
                        for j,v in enumerate(times([g for g,_ in gs])):vs[j].append(v-empty)
                    row(t,'gate_apply','neighbour',labels,vs,dict(agreement=max(diagnostics),gate='full-layer TASK required; agreement is diagnostic only'))
    except Exception as error:
        import traceback
        failure=str((t,role,phase))+': '+type(error).__name__+": "+str(error)+"; "+" | ".join(traceback.format_exc().splitlines()[-5:])
    complete=sum(r['kind']=='producer' for r in records)
    archive=dict(records=records,base_prepare_s=base_prepare,added_prepare=bank['times'],preflight_s=preflight_s,seeds=list(seeds),unit_us=.008,empty_us=empty,elapsed_s=time.monotonic()-begin,complete=complete,failure=failure)
    if callable(globals().get('DA_LOCAL_ARCHIVE')):DA_LOCAL_ARCHIVE(archive)
    encoded=da_encode(archive)
    lines=['META dense-all:36shapes,4small/7wide arms,8seeds8cold,A:A;GPU pack;actual S18 producers and neighbours;gate overlap','DATA '+encoded,'END complete='+str(complete)+' elapsed='+str(round(time.monotonic()-begin,1))]
    assert sum(len(x)+1 for x in lines)<=9000,('archive exceeds receipt cap',len(encoded));return lines
