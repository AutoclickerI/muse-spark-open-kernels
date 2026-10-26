def ls_load():
    path='<P10 large split>';linecache.cache[path]=(len(LS_BASE),None,LS_BASE.splitlines(True),path);m=types.ModuleType('large_split');m.__file__=path;exec(compile(LS_BASE,path,'exec'),m.__dict__)
    if globals().get('LS_LOCAL_INSTALL'):LS_LOCAL_INSTALL(m)
    return m.__dict__

def lab(deadline,weights,state=None):
    import contextlib,io,gc,traceback
    start=time.monotonic();m=ls_load();task={};check={}
    for i,s in enumerate(LS_TASK):exec(compile(s,'<published layer '+str(i)+'>','exec'),task)
    exec(compile(LS_CHECK,'<published projection TASK>','exec'),check)
    weights=task['generate_weights'](15000311,router_seed=15000313);log=io.StringIO();t0=time.monotonic()
    with contextlib.redirect_stdout(log):st=m['prepare'](weights)
    base_prepare=time.monotonic()-t0;t0=time.monotonic();bank=m['ls_prepare'](m,weights);prepare_s=time.monotonic()-t0
    resources={mode:[mod.resource(i,0) for i in (0,1)] for mode,mod in bank['modules'].items()};rows=[];failure=None;seeds=[15000319+104729*i for i in range(8)]
    def materialize(pair):
        p,s=pair
        return p.sum(0).bfloat16() if s>1 else p[0].bfloat16()
    def capture(fn,x,hot=False):
        scratch=torch.empty(triton.cdiv(x.numel(),1024),device='cuda');a,b=torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)
        fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            m['_ls_warm'][(scratch.numel(),)](x,scratch,x.numel(),hot,1024,num_warps=4)
            a.record();out=fn();b.record()
        g.replay();torch.cuda.synchronize();a.elapsed_time(b);return g,out,a,b
    def times(graphs):
        values=[[] for _ in graphs]
        for r in range(8):
            for j in (range(len(graphs)) if r%2==0 else range(len(graphs)-1,-1,-1)):
                lab_cold_l2();g,o,a,b=graphs[j];g.replay();torch.cuda.synchronize();values[j].append(a.elapsed_time(b)*1000)
        return [statistics.median(v) for v in values]
    try:
        for t in (128,256):
            for role in ('qkv_gate','o','pool_router','unpooler'):
                if time.monotonic()>deadline-25:raise TimeoutError('before '+str((t,role)))
                w=bank['roles'][role];n,k=w.shape;x=torch.randn(t,k,device='cuda',dtype=torch.bfloat16);res=torch.randn(t,8192,device='cuda');plan=st['best7_plans'][t//8,8,2048];bn,s=m['LS_CONFIG'][t,role];idx=0 if bn==128 else 1;normal=resources['raw'][idx];free=resources['mathfree'][idx];pad=max(free[3],free[5]//(normal[4]+1)+1024)
                matched=bank['modules']['mathfree'].resource(idx,pad);assert matched[4]==normal[4],('occupancy',t,role,normal,free,matched)
                pads={};matched_modes={}
                for mode in m['LS_MODES']:
                    rr=resources[mode][idx];pp=0 if rr[4]==normal[4] else max(rr[3],rr[5]//(normal[4]+1)+1024);rr=bank['modules'][mode].resource(idx,pp);assert rr[4]==normal[4],('factorial occupancy',t,role,mode,normal,rr);pads[mode]=pp;matched_modes[mode]=rr
                data=dict(local_start=torch.zeros((),device='cuda',dtype=torch.int32))
                labels=['S19','AA','S19_hotX','RAW','RAW_hotX','X_smem','NO_MMA','NO_MMA_occ','NO_EP','BF16_EP','S19_cast','RAW_cast','X_NO_MMA','X_NO_EP','NO_MMA_NO_EP','X_NO_MMA_NO_EP']
                def run(label):
                    if label.startswith('S19') or label=='AA':return m['_best7_project'](x,role,st,plan,torch.float32 if role=='pool_router' else torch.bfloat16)
                    mode={'X_smem':'smem_x','NO_MMA':'mathfree','NO_MMA_occ':'mathfree','NO_EP':'no_epilogue','BF16_EP':'bf16','X_NO_MMA':'x_mathfree','X_NO_EP':'x_no_ep','NO_MMA_NO_EP':'mathfree_no_ep','X_NO_MMA_NO_EP':'x_mathfree_no_ep'}.get(label,'raw')
                    return m['ls_run'](x,role,bank,mode,0 if label=='NO_MMA' else pads[mode])
                def value(label):return materialize(run(label)) if label.endswith('_cast') else run(label)
                graphs=[capture(lambda label=label:value(label),x,label.endswith('hotX')) for label in labels]
                client=role!='qkv_gate'
                def consume(pair):return m['_best7_route'](pair,st,data,plan) if role=='pool_router' else m['_best7_boundary'](pair,res,st,plan,'b1' if role=='o' else 'b2')
                legal=['S19','AA','S19_hotX','RAW','RAW_hotX']+(['BF16_EP'] if role!='pool_router' else [])
                client_graphs=[capture(lambda label=label:consume(run(label)),x,label.endswith('hotX')) for label in legal] if client else []
                vals=[[] for _ in graphs];chains=[[] for _ in client_graphs];errors={a:0. for a in ('S19','RAW','BF16_EP')}
                for seed in seeds:
                    gen=torch.Generator(device='cuda').manual_seed(seed);x.copy_(torch.randn(x.shape,device='cuda',generator=gen).bfloat16());res.copy_(torch.randn(res.shape,device='cuda',generator=gen));d=dict(x=x,config=dict(layer='qkv_gate',m=t,n=n,k=k));wt=dict(qkv_gate=w)
                    for label in ('S19','RAW','BF16_EP'):
                        pair=run(label);out=materialize(pair);ok,msg=check['check_implementation'](d,out,wt);assert ok,(t,role,label,seed,msg);errors[label]=max(errors[label],check['relative_l2'](out,check['ref_kernel'](d,wt)))
                        if role=='pool_router' and label!='BF16_EP':
                            p=pair[0].sum(0);router=p[:,4096:4352]+p[:,4352:];rd=dict(x=x,config=dict(layer='router',m=t,n=256,k=k));ok,msg=check['check_implementation'](rd,router,weights);assert ok,('router',t,label,seed,msg)
                    for j,v in enumerate(times(graphs)):vals[j].append(round(v*125))
                    if client:
                        for j,v in enumerate(times(client_graphs)):chains[j].append(round(v*125))
                    x.zero_();x[:,seed%k]=16
                    for label in ('S19','RAW','BF16_EP'):
                        ok,msg=check['check_implementation'](d,materialize(run(label)),wt);assert ok,(t,role,label,'column',msg)
                rows.append(dict(t=t,role=role,bn=bn,s=s,labels=labels,producer=vals,chain_labels=legal if client else [],neighbour=chains,occupancy=dict(raw=normal,mathfree=free,matched=matched,padding=pad,factorial_padding=pads,matched_modes=matched_modes),errors=errors));graphs=client_graphs=None;gc.collect();torch.cuda.empty_cache()
    except Exception as e:failure=repr(e)+' | '+' | '.join(traceback.format_exc().splitlines()[-6:])
    result=dict(rows=rows,seeds=seeds,unit_us=.008,resources=resources,complete=len(rows)==8 and failure is None,failure=failure,base_prepare_s=base_prepare,oracle_prepare_s=prepare_s,build_s=bank['build_s'],pack_s=bank['pack_s'],elapsed_s=time.monotonic()-start,limits=['L2-hot excludes the prefetch kernel cost; synthetic X_smem/NO_MMA/NO_EP arms are bounds, not candidates','NO_MMA retains raw weight traffic with XOR/checksum sink; its overhead makes saved-MMA time a lower bound','X_smem reuses the initial Q real tiles, removing repeated TMA loads/barrier phases; separates their combined cost','Opaque Lt routines have no exact math-free twin; matched raw-register ablations are reported separately','BF16_EP pool/router is an output-format bound, not a router precision candidate'])
    if globals().get('LS_LOCAL_ARCHIVE'):LS_LOCAL_ARCHIVE(result)
    blob=base64.b85encode(lzma.compress(json.dumps(result,separators=(',',':')).encode())).decode();assert len(blob)<8300,len(blob)
    return ['META P10 large split:S19 vs raw register twins;8shapes8seeds8cold;hotX,smemX,matched occupancy math-free,epilogue','SPLIT_XZ85 '+blob,'END complete='+str(result['complete'])]
