def sgq_load():
    import linecache
    file='<sgq candidate>';linecache.cache[file]=(len(SGQ_SOURCE),None,SGQ_SOURCE.splitlines(True),file)
    ns={'__name__':'sgq_candidate'};exec(compile(SGQ_SOURCE,file,'exec'),ns);return ns

def lab(deadline,weights,state=None):
    import time,statistics,math,json,zlib,base64
    begin=time.monotonic();m=sgq_load();T=torch;gen=T.Generator(device='cuda').manual_seed(781273)
    task={};exec(compile(SGQ_CHECK_SOURCE,'<published projection checker>','exec'),task)
    if weights is None or 'pooler' not in weights:
        def dense(n,k):return (T.randn((n,k),device='cuda',generator=gen)/math.sqrt(k)).bfloat16()
        weights=dict(qkv_gate=dense(2304,8192),o=dense(8192,1024),pooler=dense(4096,8192),unpooler=dense(8192,4096),router=.045*T.randn((256,8192),device='cuda',generator=gen))
    st=m['prepare'](weights);bank=m['sgq_prepare'](weights)
    def capture(fn):
        fn();T.cuda.synchronize();g=T.cuda.CUDAGraph()
        with T.cuda.graph(g):out=fn()
        g.replay();T.cuda.synchronize();return g,out
    def times(gs):
        ev=[]
        for r in range(8):
            for j in (range(len(gs)) if r%2==0 else range(len(gs)-1,-1,-1)):
                lab_cold_l2();a,b=T.cuda.Event(enable_timing=True),T.cuda.Event(enable_timing=True);a.record();gs[j].replay();b.record();ev.append((j,a,b))
        T.cuda.synchronize();v=[[] for _ in gs]
        for j,a,b in ev:v[j].append(a.elapsed_time(b)*1000)
        return [statistics.median(x) for x in v]
    empty=times([capture(lambda:None)[0]])[0];records=[]
    for t in (8,16):
        for n,role in ((2304,'qkv_gate'),(1280,'qkv3')):
            if time.monotonic()>deadline-20:return ['INCOMPLETE QKV dispatch, no complete batch']
            x=T.randn((t,8192),device='cuda',dtype=T.bfloat16);w=weights['qkv_gate'][:n].contiguous();st[role]=w
            def producer(arm):
                if arm>=0:return m['sgq_project'](x,n,arm,bank)
                return m['_project_variant'](x,w,role,st,m['PROJ_PINS'].get((t,role),'custom'),T.bfloat16)
            # Include the actual materialization consumed by QKV's norms. The
            # independent gate/attention stream is unchanged in the P4 patch.
            arms=[-1,-1]+list(range(len(m['SGQ_CONFIGS'])))
            gs=[capture(lambda arm=arm:m['_p12_materialize'](producer(arm))) for arm in arms]
            vals=[[] for _ in gs];errors=[0.]*len(gs);probes=[0.]*len(gs)
            for seed in (90389,21191,47309,65827,10253,85703,32783,71429,54367,16693,98627,40237,63487,28279,77617,11939):
                gen.manual_seed(seed);x.copy_(T.randn(x.shape,device='cuda',generator=gen).bfloat16());data=dict(x=x,config=dict(layer='qkv_gate',n=n,m=t,k=8192));ref=task['ref_kernel'](data,weights)
                for j,(g,out) in enumerate(gs):
                    g.replay();T.cuda.synchronize();ok,msg=task['check_implementation'](data,out,weights);assert ok,(t,role,j,seed,msg);errors[j]=max(errors[j],task['relative_l2'](out,ref))
                for j,v in enumerate(times([g for g,_ in gs])):vals[j].append(v-empty)
                # One-hot values are checked after timing, never used to select work.
                x.zero_();x[:,(seed*131)%8192]=16;ref=task['ref_kernel'](data,weights)
                for j,(g,out) in enumerate(gs):
                    g.replay();T.cuda.synchronize();ok,msg=task['check_implementation'](data,out,weights);assert ok,(t,role,j,seed,'column',msg);probes[j]=max(probes[j],task['relative_l2'](out,ref))
            for j,arm in enumerate(arms):
                p=bank['plans'][t,n,arm] if arm>=0 else None
                records.append(dict(t=t,role=role,arm='A' if j==0 else 'AA' if j==1 else arm,cfg=list(m['SGQ_CONFIGS'][arm]) if arm>=0 else None,us=vals[j],error=errors[j],probe=probes[j],weight_bytes=p['wp'].numel()*p['wp'].element_size()+p['esc'].numel()*p['esc'].element_size() if p else None))
    archive=base64.b85encode(zlib.compress(json.dumps(records,separators=(',',':')).encode(),9)).decode()
    return ['META exact12 QKV dispatch;6 geometry arms,4shapes,16new dense seeds8cold; onehot after timing; native stream order and FP32 outputs retained','DATA '+archive,'END complete=24/24 setup_and_lab_s='+str(round(time.monotonic()-begin,1))]
