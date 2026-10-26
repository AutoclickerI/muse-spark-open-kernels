def point_lab(deadline,context):
    import linecache,time,statistics
    m,weights,roles,ctrl,banks=context;T=torch;begin=time.monotonic();native=T.cuda.get_device_capability()==(9,0)
    names=('_dc_gb_kernel','_dc_gi_kernel','_dc_gemv_kernel','_dc_gate_split_kernel','_dc_vdsk_gemm')
    scalar={n:m[n] for n in names};file='<p10-grouped-point>'
    linecache.cache[file]=(len(BLOCK_SOURCE),None,BLOCK_SOURCE.splitlines(True),file);exec(compile(BLOCK_SOURCE,file,'exec'),m)
    grouped={n:m[n] for n in names}
    for name in ('qkv3','gate','o','unpooler','unpooler_slice'):
        banks[name]=m['dc_encode_cd'](m['cd_encode'](roles[name]))[0]
    m['_gate_split_prepare'](ctrl,weights['o'].device)
    st=dict(control=ctrl,roles=roles,sparse=banks,native=native)
    if native or m['DC_FORCE']:m['dc_prepare_library'](st)
    data={'local_start':T.zeros((),device='cuda',dtype=T.int32)}
    def capture(fn):
        fn();T.cuda.synchronize();graph=T.cuda.CUDAGraph()
        with T.cuda.graph(graph):out=fn()
        graph.replay();T.cuda.synchronize();return graph,out
    def times(graphs):
        events=[]
        for r in range(8):
            order=range(4) if r%2==0 else range(3,-1,-1)
            for j in order:
                lab_cold_l2();a,b=T.cuda.Event(enable_timing=True),T.cuda.Event(enable_timing=True);a.record();graphs[j].replay();b.record();events.append((j,a,b))
        T.cuda.synchronize();v=[[] for _ in range(4)]
        for j,a,b in events:v[j].append(a.elapsed_time(b)*1000)
        return [statistics.median(x) for x in v]
    empty=capture(lambda:None)[0];lab_cold_l2();a,b=T.cuda.Event(enable_timing=True),T.cuda.Event(enable_timing=True);a.record();empty.replay();b.record();T.cuda.synchronize();overhead=a.elapsed_time(b)*1000
    cases=[(8,'qkv_gate'),(16,'qkv_gate'),(16,'qkv3'),(8,'pool_router'),(16,'pool_router'),(32,'pool_router'),(1,'unpooler_slice'),(8,'unpooler_slice')]
    cases += [(t,'gate') for t in (1,8,16,32)]
    cases += [(t,n) for t in (128,256) for n in ('qkv_gate','o','pool_router','unpooler')]
    lines=['POINT_META grouped unit reader vs scalar-point reader and S15,8seeds8cold orders; native small-M geometry/reductions and gate ordering unchanged; large group arm is direct VDSK']
    done=0
    for t,name in cases:
        if time.monotonic()>deadline-15:return lines+[f'POINT_INCOMPLETE {done}/20']
        w=roles[name];x=T.randn((t,w.shape[1]),device='cuda',dtype=T.bfloat16);y=T.randn((t,1024),device='cuda',dtype=T.bfloat16)
        def produce(arm):
            if arm:
                m.update(scalar if arm==1 else grouped)
                if name=='gate':return m['dc_gate_apply'](x,y,banks[name])
                if t>=128:return m['dc_vdsk_project'](x,banks[name],(t,name)!=(128,'qkv_gate'))
                return m['dc_triton_project'](x,banks[name],name)
            if name=='gate':
                bn,s,ns=m['GATE_SPLIT_CFG'];p,cnt=m['_gate_split_buffers'](x.device,t);out=T.empty_like(y)
                m['_gate_split_kernel'][(16,s)](x,w,p,cnt,y,out,t,x.stride(0),8192,s,N=1024,BN=bn,BK=128,TB=max(16,t),NS=ns,num_warps=4,launch_pdl=True)
                return out
            if t>=128 and not native and not m['DC_FORCE']:return T.nn.functional.linear(x,w).unsqueeze(0),1
            return m['_nf_port_base_project'](x,w,name,ctrl,m['PROJ_PINS'].get((t,name),'custom'),T.float32 if name=='pool_router' else T.bfloat16)
        def consume(arm):
            pair=produce(arm)
            if name=='gate':return pair
            if name=='pool_router':
                return m['_glue_cl_route'](pair,ctrl,data,('cl',8,256,0)) if native and t<=32 else m['route_and_norm'](pair,ctrl,data)
            return m['_materialize'](pair)
        saved=[capture(lambda:consume(0)),capture(lambda:consume(0)),capture(lambda:consume(1)),capture(lambda:consume(2))]
        vals=[[] for _ in saved];err=0.;probe=0.
        for seed in (61791,14357,92003,17453,51613,72577,35111,81901):
            gen=T.Generator(device='cuda').manual_seed(seed);x.copy_(T.randn(x.shape,device='cuda',generator=gen).bfloat16())
            ref=x.double()@w.double().T
            if name=='gate':ref=y.double()*T.sigmoid(ref)
            for arm in (0,1,2):
                pair=produce(arm);got=pair if name=='gate' else m['_materialize'](pair);e=float((got.double()-ref).norm()/ref.norm())
                assert e<(.005 if got.dtype==T.bfloat16 else 3e-5),('POINT random',t,name,arm,e);err=max(err,e)
            for graph,_ in saved:graph.replay();T.cuda.synchronize()
            if name=='pool_router':
                a=saved[0][1]
                for _,b in saved[2:]:assert T.equal(a[1],b[1]) and T.equal(a[3],b[3]) and float((a[2]-b[2]).norm()/a[2].norm())<1e-4,('POINT route',t)
            original=x.clone();x.zero_();x[:,(seed*131+t)%x.shape[1]]=16;ref=x.double()@w.double().T
            if name=='gate':ref=y.double()*T.sigmoid(ref)
            for arm in (0,1,2):
                pair=produce(arm);got=pair if name=='gate' else m['_materialize'](pair);e=float((got.double()-ref).norm()/ref.norm())
                assert e<(.005 if name=='gate' else 1e-5),('POINT probe',t,name,arm,e);probe=max(probe,e)
            x.copy_(original)
            for j,v in enumerate(times([a[0] for a in saved])):vals[j].append(v-overhead)
        aa=max(abs(b/a-1) for a,b in zip(vals[0],vals[1]));rs=[b/a for a,b in zip(vals[0],vals[2])];rg=[b/a for a,b in zip(vals[0],vals[3])]
        lines.append(f'POINT {t}:{name} A={statistics.median(vals[0]):.2f} scalar={statistics.median(vals[2]):.2f} group={statistics.median(vals[3]):.2f} rS={statistics.median(rs):.4f} rG={statistics.median(rg):.4f}[{min(rg):.4f},{max(rg):.4f}] aa={aa:.4f} e={err:.1e} p={probe:.1e}')
        done+=1
    return lines+[f'POINT_END complete={done}/20 elapsed={time.monotonic()-begin:.1f}s']
