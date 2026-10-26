def follow_load():
    import linecache
    file='<sg_followup>';linecache.cache[file]=(len(FOLLOW_SOURCE),None,FOLLOW_SOURCE.splitlines(True),file);ns={'__name__':'sg_followup'};exec(compile(FOLLOW_SOURCE,file,'exec'),ns);return ns


def lab(deadline,weights,state=None):
    import statistics,time,math,json,zlib,base64
    begin=time.monotonic();m=follow_load();T=torch;gen=T.Generator(device='cuda').manual_seed(61791)
    def dense(n,k):return (T.randn((n,k),device='cuda',generator=gen)/math.sqrt(k)).bfloat16()
    if weights is None or 'pooler' not in weights:weights=dict(qkv_gate=dense(2304,8192),o=dense(8192,1024),pooler=dense(4096,8192),unpooler=dense(8192,4096),router=.045*T.randn((256,8192),device='cuda',generator=gen))
    st=m['prepare'](weights)
    for j in (1,2):
        st['alpha'+str(j)]=T.sigmoid(T.randn(8192,device='cuda',generator=gen));st['beta'+str(j)]=T.sigmoid(T.randn(8192,device='cuda',generator=gen))
    st['pre_ffn_w']=1+.1*T.randn(8192,device='cuda',generator=gen);st['next_pre_attn_w']=1+.1*T.randn(8192,device='cuda',generator=gen);st['post_ffn_w']=1+.1*T.randn(8192,device='cuda',generator=gen)
    hi=weights['router'].bfloat16();st['pool_router_fused']=T.cat((weights['pooler'],hi,(weights['router']-hi.float()).bfloat16()))
    st['router_bias']=.055*T.randn(256,device='cuda',generator=gen);st['pre_expert_w']=1+.1*T.randn(4096,device='cuda',generator=gen);data=dict(local_start=T.zeros((),device='cuda',dtype=T.int32))
    def capture(fn):
        for _ in range(2):fn();T.cuda.synchronize()
        g=T.cuda.CUDAGraph()
        with T.cuda.graph(g):out=fn()
        g.replay();T.cuda.synchronize();return g,out
    def times(graphs):
        events=[]
        for r in range(8):
            for j in (range(len(graphs)) if r%2==0 else range(len(graphs)-1,-1,-1)):
                lab_cold_l2();a,b=T.cuda.Event(enable_timing=True),T.cuda.Event(enable_timing=True);a.record();graphs[j].replay();b.record();events.append((j,a,b))
        T.cuda.synchronize();v=[[] for _ in graphs]
        for j,a,b in events:v[j].append(a.elapsed_time(b)*1000)
        return [statistics.median(x) for x in v]
    def ci(rr):
        logs=[math.log(r) for r in rr];v=statistics.mean(logs);e=2.13145*statistics.stdev(logs)/4
        return [math.exp(v),math.exp(v-e),math.exp(v+e)]
    empty=times([capture(lambda:None)[0]])[0];records=[]
    # Real dense M1 inputs, with one-hot checks outside every timed sample.
    x=T.randn((1,1024),device='cuda',dtype=T.bfloat16);res=T.randn((1,8192),device='cuda');plan=dict(glue='best6',geometry=dict(b1=('cl',8,256,1)))
    def o_control():
        pair=m['_project_variant'](x,st['o'],'o',st,'custom',T.bfloat16)
        return m['_best7_boundary'](pair,res,st,plan,'b1')
    gs=[capture(o_control),capture(o_control)]
    for wr in (4,2,1):gs.append(capture(lambda wr=wr:m['sgo_project'](x,res,st,wr)[1]))
    vals=[[] for _ in gs];errors=[0.]*len(gs)
    for seed in (68137,29009,41879,73939,12517,94421,50207,86269,39163,79579,58337,18719,96517,34781,62507,81649):
        gen.manual_seed(seed);x.copy_(T.randn(x.shape,device='cuda',generator=gen).bfloat16());res.copy_(T.randn(res.shape,device='cuda',generator=gen));ref=x.double()@weights['o'].double().T
        ss=st['alpha1'].double()*res.double()+st['beta1'].double()*ref*T.rsqrt((ref*ref).mean(-1,keepdim=True)+1e-8);hh=ss*T.rsqrt((ss*ss).mean(-1,keepdim=True)+1e-5)*st['pre_ffn_w'].double()
        for j,(g,(h,s)) in enumerate(gs):
            g.replay();T.cuda.synchronize();e=float((s.double()-ss).norm()/ss.norm());eh=float((h.double()-hh).norm()/hh.norm());assert e<.03 and eh<.03,('O1 boundary',j,seed,e,eh);errors[j]=max(errors[j],e,eh)
        for wr in (4,2,1):assert int(st['sgo']['plans'][wr]['counter'].item())==0
        for j,v in enumerate(times([g for g,_ in gs])):vals[j].append(v-empty)
    for j,label in enumerate(('A','AA','WR4','WR2','WR1')):
        records.append(dict(t=1,role='o_b1',arm=label,us=vals[j],ratio=ci([b/a for a,b in zip(vals[0],vals[j])]),error=errors[j]))
    # Current in-layer FP32 fused pool/router bank; same consumers on every arm.
    for t in (128,256):
        for role in ('qkv_gate','o','pool_router','unpooler'):
            if time.monotonic()>deadline-20:return ['INCOMPLETE projection followup; no complete batch']
            plan=dict(glue='best6',geometry=dict(b1=(16,1),b2=(8,1) if t==128 else (16,1)));w=st['pool_router_fused'] if role=='pool_router' else st[role];n,k=w.shape;x=T.randn((t,k),device='cuda',dtype=T.bfloat16);res=T.randn((t,8192),device='cuda')
            def producer(arm):
                if arm>=0:return m['sg_reuse_project'](x,w,arm)
                return m['_project_variant'](x,w,role,st,m['PROJ_PINS'].get((t,role),'custom'),T.float32 if role=='pool_router' else T.bfloat16)
            def consume(pair):
                if role=='pool_router':return m['route_and_norm'](pair,st,data)
                if role in ('o','unpooler'):return m['_best7_boundary'](pair,res,st,plan,'b1' if role=='o' else 'b2')
                return m['_p12_materialize'](pair)
            arms=[-1,-1]+list(range(len(m['SG_REUSE_CONFIGS'][t])));gs=[capture(lambda arm=arm:consume(producer(arm))) for arm in arms];vals=[[] for _ in gs];errors=[0.]*len(gs)
            for seed in (68137,29009,41879,73939,12517,94421,50207,86269,39163,79579,58337,18719,96517,34781,62507,81649):
                gen.manual_seed(seed);x.copy_(T.randn(x.shape,device='cuda',generator=gen).bfloat16());res.copy_(T.randn(res.shape,device='cuda',generator=gen));ref=x.double()@w.double().T
                for j,(g,out) in enumerate(gs):
                    g.replay();T.cuda.synchronize()
                    if role=='pool_router':e=sg_follow_route(out,x,weights['router'],st['router_bias'],st['pre_expert_w'],ref[:,:4096])
                    elif role in ('o','unpooler'):
                        a=st['alpha1' if role=='o' else 'alpha2'].double();b=st['beta1' if role=='o' else 'beta2'].double();pre=st['pre_ffn_w' if role=='o' else 'next_pre_attn_w'].double();br=ref if role=='o' else ref*st['post_ffn_w'].double()
                        ss=a*res.double()+b*br*T.rsqrt((br*br).mean(-1,keepdim=True)+1e-8);hh=ss*T.rsqrt((ss*ss).mean(-1,keepdim=True)+1e-5)*pre;e=max(float((out[0].double()-hh).norm()/hh.norm()),float((out[1].double()-ss).norm()/ss.norm()));assert e<.03,(t,role,j,e)
                    else:e=float((out.double()-ref).norm()/ref.norm());assert e<.01,(t,role,j,e)
                    errors[j]=max(errors[j],e)
                for j,v in enumerate(times([g for g,_ in gs])):vals[j].append(v-empty)
            for j,arm in enumerate(arms):records.append(dict(t=t,role=role,arm='A' if j==0 else 'AA' if j==1 else list(m['SG_REUSE_CONFIGS'][t][arm]),us=vals[j],ratio=ci([b/a for a,b in zip(vals[0],vals[j])]),error=errors[j],weight_bytes=w.numel()*w.element_size(),issued_weight_bytes=w.numel()*w.element_size() if arm>=0 else None))
    # Keep every joint seed sample for later composition estimates, not just CIs.
    archive=base64.b85encode(zlib.compress(json.dumps(records,separators=(',',':')).encode(),9)).decode()
    return ['META O1 last-CTA norm/residual vs unchanged NF O1+boundary; large BT256 reuse and split-K, same neighbours,16seeds8cold','DATA '+archive,'END complete=o1:3,large:20; gates all pass']
