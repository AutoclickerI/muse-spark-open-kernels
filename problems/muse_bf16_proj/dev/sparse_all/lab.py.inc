def dc_load_candidate():
    import linecache
    ns={'__name__':'p10_sparse_dense'};file='<p10_sparse_dense>'
    linecache.cache[file]=(len(DC_SOURCE),None,DC_SOURCE.splitlines(True),file)
    exec(compile(DC_SOURCE,file,'exec'),ns);return ns


def lab(deadline,weights,state=None):
    import time,math,statistics
    begin=time.monotonic();m=dc_load_candidate();T=torch
    if weights is None or 'pooler' not in weights:
        # Public dense distribution, independent of the reference expert RNG
        # prefix. Router includes the published mean-direction component.
        g=T.Generator(device='cuda').manual_seed(61791)
        def dense(n,k):return (T.randn((n,k),device='cuda',generator=g)/math.sqrt(k)).bfloat16()
        w=dict(qkv_gate=dense(2304,8192),o=dense(8192,1024),pooler=dense(4096,8192),unpooler=dense(8192,4096))
        mean=T.randn(8192,generator=T.Generator().manual_seed(5354)).cuda()
        w['router']=.045*T.randn((256,8192),device='cuda',generator=g)+.07*T.randn((256,1),device='cuda',generator=g)*mean/math.sqrt(8192)
        weights=w
    m['cd_module'](((1,1,2,3),));st=m['dc_prepare_roles'](weights)
    st['control']['pre_expert_w']=T.ones(4096,device='cuda');st['control']['router_bias']=T.zeros(256,device='cuda')
    data={'local_start':T.zeros((),device='cuda',dtype=T.int32)}
    def material(pair):return m['_materialize'](pair)
    def route(pair):
        if st['native'] and pair[0].shape[1]<=32:return m['_glue_cl_route'](pair,st['control'],data,('cl',8,256,0))
        return m['route_and_norm'](pair,st['control'],data)
    def capture(fn):
        for _ in range(2):
            fn();T.cuda.synchronize()
        T.cuda.synchronize();graph=T.cuda.CUDAGraph()
        with T.cuda.graph(graph):out=fn()
        graph.replay();T.cuda.synchronize();return graph,out
    def times(graphs,rounds=8):
        events=[]
        for r in range(rounds):
            order=range(len(graphs)) if r%2==0 else range(len(graphs)-1,-1,-1)
            for j in order:
                lab_cold_l2();a,b=T.cuda.Event(enable_timing=True),T.cuda.Event(enable_timing=True)
                a.record();graphs[j].replay();b.record();events.append((j,a,b))
        T.cuda.synchronize();out=[[] for _ in graphs]
        for j,a,b in events:out[j].append(a.elapsed_time(b)*1000)
        return [statistics.median(v) for v in out]
    empty=times([capture(lambda:None)[0]])[0]
    lines=[f'META all dense BF16 roles S15=60254883; sparse plane + exact escapes; setup={time.monotonic()-begin:.1f}s',
           'META 8 fresh seeds; 8 cold alternating replays each; A:A independent capture; B same structure (library includes decode); C direct coded GEMM where library blocked']
    for name,s in st['stats'].items():lines.append(f'BYTES {name} point={s["total_bytes"]} body={s["body_bytes"]} bpw={s["bits_per_weight"]:.5f}')
    for (t,name),s in sorted(st['nf_sparse']['stats'].items()):lines.append(f'NF {t}:{name} bytes={s["total_bytes"]} bpw={s["bits_per_weight"]:.5f}')
    # Every requested matrix at all six loads. qkv3 is the active gate-split
    # subset. Unpooler's M1/8 call site consumes only the first1024 output rows.
    cases=[(t,name) for t in (1,8,16,32,128,256) for name in ('qkv_gate','o','gate','pooler','unpooler','router_hi','router_lo','pool_router','router_sum')]
    cases += [(t,'qkv3') for t in (1,16,32)]
    risky={(1,'qkv3'),(32,'qkv3'),(32,'qkv_gate'),(32,'o'),(32,'unpooler'),(128,'qkv_gate')}
    cases.sort(key=lambda a:(a in risky,a[0],a[1]))
    seeds=(61791,14357,92003,17453,51613,72577,35111,81901)
    count=0;direct_count=0
    for t,name in cases:
        if time.monotonic()>deadline-15:
            return lines+[f'INCOMPLETE rows={count}/{len(cases)} elapsed={time.monotonic()-begin:.1f}; do not promote']
        role='unpooler_slice' if name=='unpooler' and t<=8 else name
        ww=weights['router'] if name=='router_sum' else st['roles'][role]
        x=T.randn((t,ww.shape[1]),device='cuda',dtype=T.bfloat16);y=T.randn((t,1024),device='cuda',dtype=T.bfloat16)
        def project(sparse,which=role,direct=False):
            return m['dc_vdsk_project'](x,st['sparse'][which],True) if direct else m['dc_project'](x,which,st,sparse)
        def call(sparse,direct=False):
            if name=='gate':
                if t<=32:
                    if sparse:return m['dc_gate_apply'](x,y,st['sparse']['gate'])
                    bn,s,ns=m['GATE_SPLIT_CFG'];p,cnt=m['_gate_split_buffers'](x.device,t);out=T.empty_like(y)
                    m['_gate_split_kernel'][(1024//bn,s)](x,st['roles']['gate'],p,cnt,y,out,t,x.stride(0),8192,s,N=1024,BN=bn,BK=128,TB=max(16,m['triton'].next_power_of_2(t)),NS=ns,num_warps=4,launch_pdl=True)
                    return out
                pair=project(sparse,direct=direct) if sparse else (T.nn.functional.linear(x,ww).unsqueeze(0),1)
                return (y.float()*T.sigmoid(material(pair).float())).bfloat16()
            if name=='router_sum':
                return material(project(sparse,'router_hi',direct)).float()+material(project(sparse,'router_lo',direct)).float()
            pair=project(sparse,direct=direct)
            return route(pair) if name=='pool_router' else material(pair)
        has_direct=t>=128 and (t,name)!=(128,'qkv_gate')
        labels=['A','AA','B']+(['C'] if has_direct else [])
        saved=[capture(lambda:call(False)),capture(lambda:call(False)),capture(lambda:call(True))]
        if has_direct:saved.append(capture(lambda:call(True,True)))
        graphs=[a[0] for a in saved];vals=[[] for _ in saved];errors=[0.]*len(saved);probes=[0.]*len(saved)
        for seed in seeds:
            gen=T.Generator(device='cuda').manual_seed(seed)
            x.copy_(T.randn(x.shape,device='cuda',generator=gen).bfloat16())
            if seed in (17453,35111):x.mul_(3)
            # Dense M1 input remains random; the separate one-hot probe below
            # covers exact decoder origins/signs/escape handling.
            if t>1:x[-1]=0.;x[-1,(seed*131+t)%x.shape[1]]=16
            ref=x.double()@ww.double().T
            if name=='gate':ref=y.double()*T.sigmoid(ref)
            for graph,_ in saved:
                graph.replay();T.cuda.synchronize()
            T.cuda.synchronize()
            if name=='pool_router':
                a=saved[0][1]
                for j,(_,b) in enumerate(saved[2:],2):
                    assert T.equal(a[1],b[1]) and T.equal(a[3],b[3]),('route ids',t,j,seed)
                    error=float((a[0].float()-b[0].float()).norm()/a[0].float().norm());assert error<.004,(name,t,j,error)
                    errw=float((a[2]-b[2]).norm()/a[2].norm());assert errw<1e-4,(name,t,j,errw)
                    errors[j]=max(errors[j],error,errw)
            else:
                for j,(_,out) in enumerate(saved):
                    error=float((out.double()-ref).norm()/ref.norm());tol=.005 if out.dtype==T.bfloat16 else 3e-5
                    assert error<tol,(name,t,j,seed,error)
                    errors[j]=max(errors[j],error)
                    if t>1 and name!='gate':
                        probe=float((out[-1].double()-ref[-1]).norm()/ref[-1].norm());assert probe<1e-5,(name,t,j,seed,probe)
                        probes[j]=max(probes[j],probe)
            if t==1 and name not in ('gate','pool_router'):
                original=x.clone();x.zero_();x[0,(seed*131)%x.shape[1]]=16;probe_ref=x.double()@ww.double().T
                for j,(_,out) in enumerate(saved):
                    graphs[j].replay();T.cuda.synchronize();probe=float((out.double()-probe_ref).norm()/probe_ref.norm());assert probe<1e-5,(name,t,j,seed,probe)
                    probes[j]=max(probes[j],probe)
                x.copy_(original)
            got=times(graphs)
            for j,v in enumerate(got):vals[j].append(v-empty)
        aa=max(abs(b/a-1) for a,b in zip(vals[0],vals[1]))
        for j in range(2,len(saved)):
            rr=[b/a for a,b in zip(vals[0],vals[j])]
            lines.append(f'R {t}:{name}:{labels[j]} {statistics.median(vals[0]):.2f} {statistics.median(vals[j]):.2f} r={statistics.median(rr):.4f}[{min(rr):.4f},{max(rr):.4f}] aa={aa:.4f} e={errors[j]:.1e} p={probes[j]:.1e}')
        count+=1;direct_count+=has_direct
        del saved,graphs
    return lines+[f'END complete={count}/{len(cases)} direct={direct_count} elapsed={time.monotonic()-begin:.1f}s']
