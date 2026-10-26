def lab(deadline,weights,state=None):
    import linecache,statistics,time,math
    begin=time.monotonic();p={'__name__':'p10_nf_sparse_adapter'};file='<nf-sparse-adapter>'
    linecache.cache[file]=(len(SP_SOURCE),None,SP_SOURCE.splitlines(True),file);exec(compile(SP_SOURCE,file,'exec'),p)
    old={'__name__':'p10_s15_pool_control'};file='<s15-pool-control>';linecache.cache[file]=(len(BASE_SOURCE),None,BASE_SOURCE.splitlines(True),file);exec(compile(BASE_SOURCE,file,'exec'),old)
    if weights is None or 'pooler' not in weights:
        g=torch.Generator(device='cuda').manual_seed(61791)
        def rand(n,k):return (torch.randn((n,k),device='cuda',generator=g)/math.sqrt(k)).bfloat16()
        weights=dict(pooler=rand(4096,8192),router=.045*torch.randn((256,8192),device='cuda',generator=g),qkv_gate=rand(2304,8192),o=rand(8192,1024),unpooler=rand(8192,4096))
    hi=weights['router'].bfloat16();w=torch.cat((weights['pooler'],hi,(weights['router']-hi.float()).bfloat16()))
    baseline_state=dict(qkv_gate=weights['qkv_gate'],pool_router=w,o=weights['o'],unpooler=weights['unpooler'],unpooler_slice=weights['unpooler'][:1024])
    old['_seed_v9_gemv_choices'](baseline_state);old['cd_module'](((1,1,2,3),));old['_cd_prepare'](baseline_state)
    baseline_state['pre_expert_w']=torch.ones(4096,device='cuda');baseline_state['router_bias']=torch.zeros(256,device='cuda')
    data={'local_start':torch.zeros((),device='cuda',dtype=torch.int32)}
    geometries=[('w1k8',1,8,1,1),('w2k8',2,8,1,1),('w1k4',1,4,1,1),('w1k4s4',1,4,4,2)]
    plans={};stats=[]
    for label,wr,wk,s,st in geometries:
        for sparse,vector in ((False,False),(True,False),(True,True)):
            name=label+('vec' if vector else 'sp' if sparse else 'raw');bank=p['sp_prepare'](weights,wr,wk,s,st,sparse,vector);plans[name]=bank
            stats.append((name,bank['stats']))
    def graph(fn):
        for _ in range(2):fn();torch.cuda.synchronize()
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):fn()
        g.replay();torch.cuda.synchronize();return g
    def times(graphs,rounds):
        events=[]
        for r in range(rounds):
            order=list(range(len(graphs)));order=order if r%2==0 else order[::-1]
            for j in order:
                lab_cold_l2();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();graphs[j].replay();b.record();events.append((j,a,b))
        torch.cuda.synchronize();vals=[[] for _ in graphs]
        for j,a,b in events:vals[j].append(a.elapsed_time(b)*1000)
        return [statistics.median(v) for v in vals]
    empty=times([graph(lambda:None)],8)[0]
    lines=[f'META P10 fused pool_router on attributed NF structure; all12 geometry/codec arms x8 seeds x12 cold ABBA; setup={time.monotonic()-begin:.1f}s',
        'META byte size includes11fixed bits + presence/prefix + sparse uint32; exact original escape lists identical; S15 coded/bal2 controls and route/norm consumer']
    for name,st in stats:lines.append(f'BYTES {name} packed={st["packed_bytes"]} bits={st["bits_per_weight"]:.5f}')
    native=torch.cuda.get_device_capability()==(9,0)
    for t in (1,8,16,32):
        if time.monotonic()>deadline-10:raise RuntimeError('incomplete pool/sparse lab')
        x=torch.randn((t,8192),device='cuda',dtype=torch.bfloat16)
        def control():return old['_project_variant'](x,w,'pool_router',baseline_state,'custom',torch.float32)
        def consume(pair):return old['_glue_cl_route'](pair,baseline_state,data,('cl',8,256,0)) if native else old['route_and_norm'](pair,baseline_state,data)
        if native and t==16:
            for name,bank in plans.items():
                pp=bank['plan'];idx=next(i for i,v in enumerate(bank['insts']) if v[1]==2);out=torch.empty((t,4608),device='cuda')
                args=(idx,pp['wp'].data_ptr(),x.data_ptr(),out.data_ptr(),pp['esc'].data_ptr(),t,4608,8192,*pp['cfg'],pp['t01'],pp['u01'],torch.cuda.current_stream().cuda_stream)
                lines.append(f'INFO {name} regs_local={pp["fn"](*args,2)} active={pp["fn"](*args,1)}')
        names=['S15','AA']+list(plans);graphs=[graph(lambda:consume(control())),graph(lambda:consume(control()))]
        for name in plans:graphs.append(graph(lambda name=name:consume(p['sp_project'](x,plans[name]))))
        rows={n:[] for n in names};maxerr={n:0. for n in plans};probes={n:0. for n in plans}
        for seed in (61791,14357,92003,17453,51613,72577,35111,81901):
            g=torch.Generator(device='cuda').manual_seed(seed);x.copy_(torch.randn(x.shape,device='cuda',generator=g).bfloat16())
            if t>1:x[-1]=0.;x[-1,(seed*131+t)%8192]=16.
            ref=x.double()@w.double().T
            cp=control();cv=cp[0].double().sum(0);ce=float((cv-ref).norm()/ref.norm());assert ce<3e-5,('CONTROL',t,seed,ce)
            route_a=consume(cp);torch.cuda.synchronize()
            for name,bank in plans.items():
                got=p['sp_project'](x,bank)[0][0].double();err=float((got-ref).norm()/ref.norm());probe=float((got[-1]-ref[-1]).norm()/ref[-1].norm())
                assert err<3e-5 and (t==1 or probe<1e-5),(t,name,err,probe);maxerr[name]=max(maxerr[name],err)
                route_b=consume(p['sp_project'](x,bank));torch.cuda.synchronize()
                assert torch.equal(route_a[1],route_b[1]) and torch.equal(route_a[3],route_b[3]),('route ids',t,name,seed)
                assert float((route_a[2]-route_b[2]).norm()/route_a[2].norm())<1e-4,('route weights',t,name,seed)
                assert float((route_a[0].float()-route_b[0].float()).norm()/route_a[0].float().norm())<.004,('pooled norm',t,name,seed)
            original=x.clone();x.zero_();x[:,(seed*131+t)%8192]=16.;probe_ref=x.double()@w.double().T
            cv=control()[0].double().sum(0);assert float((cv-probe_ref).norm()/probe_ref.norm())<1e-5,('CONTROL probe',t,seed)
            for name,bank in plans.items():
                got=p['sp_project'](x,bank)[0][0].double();probe=float((got-probe_ref).norm()/probe_ref.norm());assert probe<1e-5,(t,name,'probe',probe);probes[name]=max(probes[name],probe)
            x.copy_(original)
            vals=times(graphs,12)
            for name,v in zip(names,vals):rows[name].append(v-empty)
        aa=max(abs(b/a-1) for a,b in zip(rows['S15'],rows['AA']))
        for name in plans:
            rr=[b/a for a,b in zip(rows['S15'],rows[name])]
            lines.append(f'R t={t} {name} A={statistics.mean(rows["S15"]):.2f} B={statistics.mean(rows[name]):.2f} r={statistics.mean(rr):.5f}[{min(rr):.5f},{max(rr):.5f}] aa={aa:.5f} e={maxerr[name]:.1e} p={probes[name]:.1e}')
        del graphs
    return lines+[f'END elapsed={time.monotonic()-begin:.1f}s']
